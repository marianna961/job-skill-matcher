import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_dir = "../ner_parser/ner_model"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def extract_entities(text):
    results = ner_pipeline(text)
    results = [result for result in results if result["score"] > 0.05]

    entities = {}
    for result in results:
        entity_group = result["entity_group"]
        word = result["word"]

        if entity_group in entities:
            entities[entity_group] += " " + word
        else:
            entities[entity_group] = word

    return entities