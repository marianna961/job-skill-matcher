"""
скрипт для объедения лейбов PROFESSION и SKILLS
"""
import json

with open('upd_NER.json', 'r', encoding='utf-8') as f:
    ner_data = json.load(f)

label_mapping = {
    "B-PROFESSION": "B-PROFESSION_SKILLS",
    "I-PROFESSION": "I-PROFESSION_SKILLS",
    "B-SKILLS": "B-PROFESSION_SKILLS",
    "I-SKILLS": "I-PROFESSION_SKILLS"
}

for item in ner_data:
    item["ner_tags"] = [label_mapping.get(tag, tag) for tag in item["ner_tags"]]

with open('updated_ner_data.json', 'w', encoding='utf-8') as f:
    json.dump(ner_data, f, ensure_ascii=False, indent=4)

print("готово")