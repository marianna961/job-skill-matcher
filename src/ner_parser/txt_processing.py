"""
скрипт для обработки txt файлов в одну строку
    Пробелы важны для токенезации!
"""
import os

folder_path = r"C:\Users\maria\OneDrive\Documents\VSCode\job-skill-matcher\src\ner_parser\datasets\dataset401-420"

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        one_line_text = content.replace('\r', ' ').replace('\n', ' ').strip()

        one_line_text = ' '.join(one_line_text.split())

        #сохраняем по исходному пути
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(one_line_text)

print("done")
