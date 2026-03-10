import datasets

try:
    import transformers
    
    evaluator_name = "s-nlp/ruRoberta-large-paraphrase-v1"
    meaning_model = transformers.AutoModelForSequenceClassification.from_pretrained(evaluator_name)
    meaning_tokenizer = transformers.AutoTokenizer.from_pretrained(evaluator_name)
except ImportError:
    print(
        "Can not import transformers. If you try to score nerel-bench, do `pip install transformers`"
    )


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "input_task": doc["instruction"].format(**doc["context"]),
            "true_answer": doc["true_answer"],
        }
        return out_doc

    return dataset.map(_process_doc)


def definition_f1_score(prediction: str, ground_truth: str) -> float:
    with torch.inference_mode():
        batch = meaning_tokenizer(
            prediction, ground_truth, 
            truncation=True, max_length=meaning_model.config.max_position_embeddings, return_tensors='pt',
        ).to(meaning_model.device)
        proba = torch.softmax(meaning_model(**batch).logits, -1)
    return proba[0][1].item()


def entity_f1_score(prediction: str, ground_truth: str) -> float:
    predicted_entities = set(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), prediction.split("\n"))))
    reference_entities = set(filter(lambda it2: len(it2) > 0, map(lambda it1: it1.strip(), ground_truth.split("\n"))))
    common = predicted_entities & reference_entities
    num_same = len(common)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(predicted_entities)
    recall = 1.0 * num_same / len(reference_entities)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

