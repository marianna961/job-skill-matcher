import os
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

"""
Тест ner_model на реальных файлах с сохранением результатов в CSV.
"""

# Шаг 1: Загрузка модели и токенизатора
model_dir = "./ner_model"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

# Создаём pipeline NER с агрегацией
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Шаг 2: Чтение резюме из файла
def read_resume(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# Шаг 3: Применение NER-модели

def extract_entities(text):
    print("Запуск extract_entities")

    results = ner_pipeline(text)
    print("NER выполнен, количество результатов:", len(results))
    results = [r for r in results if r["score"] > 0.05]

    print("Пример результата:")
    for r in results[:5]:
        print(r)

    entities = {}
    current_entity = None
    current_text = ""

    for res in results:
        word = res["word"]
        label = res["entity"]

        # Начало новой сущности
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

    # Добавляем последнюю сущность
    if current_entity:
        if current_entity in entities:
            entities[current_entity].append(current_text.strip())
        else:
            entities[current_entity] = [current_text.strip()]

    # Объединяем списки сущностей в одну строку (если нужно)
    for key in entities:
        entities[key] = "; ".join(entities[key])

    return entities

# Шаг 4: Обработка всех файлов в директории
def process_resumes(input_dir, output_file):
    # Список для хранения данных всех резюме
    all_data = []
    
    # Проходим по всем файлам в директории
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        
        # Пропускаем не-txt файлы
        if not file_name.endswith(".txt"):
            continue
        
        # Читаем текст резюме
        resume_text = read_resume(file_path)
        
        # Извлекаем сущности
        entities = extract_entities(resume_text)
        
        # Добавляем имя файла (человека) к данным
        entities["person"] = file_name
        
        # Добавляем данные в общий список
        all_data.append(entities)
    
    # Преобразуем данные в DataFrame
    df = pd.DataFrame(all_data)
    
    # Устанавливаем столбец "person" первым
    if "person" in df.columns:
        df = df.set_index("person").reset_index()
    
    # Сохраняем данные в CSV
    df.to_csv(output_file, encoding="utf-8-sig", index=False)
    print(f"Результат сохранен в файл: {output_file}")

if __name__ == "__main__":
    # Директория с входными файлами
    input_dir = r"C:\Users\maria\OneDrive\Documents\VSCode\job-skill-matcher\src\ner_parser\test_resume_dataset"
    
    # Выходной CSV-файл
    output_file = "output.csv"
    
    # Обработка всех резюме
    process_resumes(input_dir, output_file)
