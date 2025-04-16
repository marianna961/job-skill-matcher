"""
заменять у ["Знание", "языков", "Русский", "—", "Родной"] лейблы на "O"
"""
import json

with open('datasets_labeled_converted/upd_NER.json', 'r', encoding='utf-8') as f:
    ner_data = json.load(f)


words_to_replace = ["Знание", "языков", "Русский", "—", "Родной"]

# Функция для замены меток
def replace_labels(data, words):
    for item in data:
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]
        
        for i, token in enumerate(tokens):
            if token in words:
                ner_tags[i] = "O"
    
    return data

updated_data = replace_labels(ner_data, words_to_replace)

with open('datasets_labeled_converted/upd_NER.json', 'w', encoding='utf-8') as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4)

print("saved")