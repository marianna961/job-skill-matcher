import json
import os
from transformers import pipeline

"""
тест ner_model на реальных файлах
"""
# Шаг 1
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_dir = "./ner_model"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

# Создаём pipeline NER с агрегацией
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Шаг 2
def read_resume(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Шаг 3: Применение NER-модели
def extract_entities(text):
    # Получаем результаты от модели
    results = ner_pipeline(text)
    
    # Фильтруем результаты (опционально, если нужно убрать низкокачественные предсказания)
    results = [result for result in results if result["score"] > 0.25]
    
    # Преобразуем результаты в удобный формат
    entities = [
        {
            "entity_group": result["entity_group"],
            "text": result["word"],
            "start": result["start"],
            "end": result["end"]
        }
        for result in results
    ]
    return entities

# Шаг 4: Сохранение данных в JSON
def save_to_json(data, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Шаг 5: Основная функция
def process_resume(input_file, output_file):

    resume_text = read_resume(input_file)
    
    entities = extract_entities(resume_text)
    
    resume_data = {
        "resume": {
            "text": resume_text,
            "entities": entities
        }
    }
    
    save_to_json(resume_data, output_file)
    print(f"Результат сохранен в файл: {output_file}")

if __name__ == "__main__":
    input_file = r"C:\Users\maria\OneDrive\Documents\VSCode\job-skill-matcher\src\ner_parser\test_resume_dataset\resume2.txt"
    output_file = "output.json"
    
    process_resume(input_file, output_file)

