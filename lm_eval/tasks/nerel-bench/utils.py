import json
import math

import datasets

try:
    import transformers
except ImportError:
    print(
        "Can not import transformers. If you try to score nerel-bench, do `pip install transformers`"
    )
    
try:
    import torch
except ImportError:
    print(
        "Can not import torch. If you try to score nerel-bench, do `pip install torch`"
    )
    
try:
    import numpy as np
except ImportError:
    print(
        "Can not import numpy. If you try to score nerel-bench, do `pip install numpy`"
    )


evaluator_name = "s-nlp/ruRoberta-large-paraphrase-v1"
meaning_model = transformers.AutoModelForSequenceClassification.from_pretrained(
    evaluator_name,
    device_map="auto"
)
meaning_tokenizer = transformers.AutoTokenizer.from_pretrained(evaluator_name)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "input_task": doc["instruction"].format(**json.loads(doc["context"])),
            "true_answer": doc["true_answer"],
        }
        return out_doc

    return dataset.map(_process_doc)


def similarity(predictions, references, **kwargs) -> float:
    """
    Calculate semantic similarity between prediction and reference
    using a paraphrase detection model.
    
    Args:
        predictions: List of predicted strings (length 1 for single prediction)
        references: List of reference strings (length 1 for single reference)
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        float: Similarity score between 0 and 1
    """
    num_samples = len(predictions)
    if num_samples != len(references):
        raise ValueError(f'The predictions do not correspond to the references! {num_samples} != {len(references)}')
    if 'minibatch' in kwargs:
        minibatch = int(kwargs['minibatch'])
        if minibatch < 1:
            raise ValueError(f"The minibatch is wrong! Expected a positive integer, got {kwargs['minibatch']}.")
    else:
        minibatch = 1
    num_batches = math.ceil(num_samples / minibatch)
    scores = []
    for batch_idx in range(num_batches):
        batch_start = batch_idx * minibatch
        batch_end = min(num_samples, batch_start + minibatch)
        with torch.inference_mode():
            batch = meaning_tokenizer(
                predictions[batch_start:batch_end], references[batch_start:batch_end], 
                truncation=True, max_length=meaning_model.config.max_position_embeddings, return_tensors='pt',
            ).to(meaning_model.device)
            proba = torch.softmax(meaning_model(**batch).logits, -1)
            scores.append(proba.to('float32').cpu().numpy().flatten())
            del proba, batch
    return float(np.mean(np.concatenate(scores)))


def f1(predictions, references, **kwargs) -> float:
    """
    Calculate F1 score for entity recognition based on line-by-line comparison.
    
    Args:
        predictions: List of predicted strings (length 1 for single prediction)
        references: List of reference strings (length 1 for single reference)
        **kwargs: Additional keyword arguments (ignored)
    
    Returns:
        float: F1 score between 0 and 1
    """
    num_samples = len(predictions)
    if num_samples != len(references):
        raise ValueError(f'The predictions do not correspond to the references! {num_samples} != {len(references)}')
    num_same = 0
    num_predicted = 0
    num_reference = 0
    for sample_idx in range(num_samples):
        predicted_entities = set(filter(
            lambda it2: len(it2) > 0,
            map(lambda it1: it1.strip(), predictions[sample_idx].split("\n"))
        ))
        reference_entities = set(filter(
            lambda it2: len(it2) > 0,
            map(lambda it1: it1.strip(), references[sample_idx].split("\n"))
        ))
        common = predicted_entities & reference_entities
        num_same += len(common)
        num_predicted += len(predicted_entities)
        num_reference += len(reference_entities)
        del predicted_entities, reference_entities, common
    if num_same > 0:
        precision = num_same / float(num_predicted)
        recall = num_same / float(num_reference)
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    return f1

