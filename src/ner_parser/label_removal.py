"""
скрипт для удаления данных с юзалес лейблами
"""

import json

with open('updated_ner_data.json', 'r', encoding='utf-8') as f:
    ner_data = json.load(f)

def filter_tokens(data):
    filtered_data = []
    for item in data:
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]
        
        filtered_tokens = [token for token, tag in zip(tokens, ner_tags) if tag not in {"I-OTHER", "B-OTHER", "I-LINKS", "B-LINKS"}]
        filtered_tags = [tag for tag in ner_tags if tag not in {"I-OTHER", "B-OTHER", "I-LINKS", "B-LINKS"}]
        
        if filtered_tokens and filtered_tags:
            filtered_data.append({
                "tokens": filtered_tokens,
                "ner_tags": filtered_tags
            })
    return filtered_data

filtered_data = filter_tokens(ner_data)

with open('without_other.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

print("label_removal")