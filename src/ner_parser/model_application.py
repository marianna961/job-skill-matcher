import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_dir = "./ner_model"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

def read_resume(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

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

def process_and_clean_resumes(input_dir, output_file):
    all_data = []

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)

        if not file_name.endswith(".txt"):
            continue

        resume_text = read_resume(file_path)
        entities = extract_entities(resume_text)
        entities["person"] = file_name
        all_data.append(entities)

    df = pd.DataFrame(all_data)

    if "person" in df.columns:
        df = df.set_index("person").reset_index()

    df = df.applymap(
        lambda x: x.replace(" ##", "")
                  .replace("##", "")
                  .replace("  ", " ")
                  .replace(" @ ", "@")
                  .replace(" . ", ".")
                  .strip()
        if isinstance(x, str) else x
    )

    df.to_csv(output_file, encoding="utf-8-sig", index=False)

if __name__ == "__main__":
    input_dir = r"C:\Users\maria\OneDrive\Documents\VSCode\job-skill-matcher\src\ner_parser\test_resume_dataset"
    output_file = "output.csv"

    process_and_clean_resumes(input_dir, output_file)
