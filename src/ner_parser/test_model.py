import os
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

"""
Тест ner_model на реальных файлах с сохранением результатов в CSV.
"""

model_dir = "./ner_model"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

def read_resume(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_entities(text):
    results = ner_pipeline(text)
    print("NER выполнен, количество результатов:", len(results))
    results = [r for r in results if r["score"] > 0.05]

    print("результат:")
    for r in results[:5]:
        print(r)

    entities = {}
    current_entity = None
    current_text = ""

    for res in results:
        word = res["word"]
        label = res["entity"]

        if label.startswith("B-"):
            if current_entity:
                if current_entity in entities:
                    entities[current_entity].append(current_text.strip())
                else:
                    entities[current_entity] = [current_text.strip()]
            current_entity = label[2:]
            current_text = word.lstrip("##")
        elif label.startswith("I-") and current_entity:
            if word.startswith("##"):
                current_text += word[2:]
            else:
                current_text += " " + word
        else:
            continue

    if current_entity:
        if current_entity in entities:
            entities[current_entity].append(current_text.strip())
        else:
            entities[current_entity] = [current_text.strip()]

    for key in entities:
        entities[key] = "; ".join(entities[key])

    return entities

def process_resumes(input_dir, output_file):
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
    
    df.to_csv(output_file, encoding="utf-8-sig", index=False)

if __name__ == "__main__":
    input_dir = r"C:\Users\maria\OneDrive\Documents\VSCode\job-skill-matcher\src\ner_parser\test_resume_dataset"
    
    output_file = "output.csv"
    
    process_resumes(input_dir, output_file)
