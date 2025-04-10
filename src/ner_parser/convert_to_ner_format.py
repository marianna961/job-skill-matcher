import json
import spacy

nlp = spacy.blank("ru")

def convert_to_ner_format(json_files):
    """
    Конвектр JSON файлов в формат NER
    """
    combined_ner_data = []

    for json_file in json_files:
        print(f"Processing file: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            text = item['data']['text']
            annotations = item['annotations'][0]['result']

            # получение позиции каждого токена
            doc = nlp(text)
            tokens = [token.text for token in doc]
            token_offsets = [(token.idx, token.idx + len(token)) for token in doc]
            labels = ['O'] * len(tokens)

            for ann in annotations:
                start = ann['value']['start']
                end = ann['value']['end']
                label = ann['value']['labels'][0]

                for i, (token_start, token_end) in enumerate(token_offsets):
                    if token_start >= start and token_end <= end:
                        if token_start == start:
                            labels[i] = f"B-{label}"
                        else:
                            labels[i] = f"I-{label}"

            combined_ner_data.append({
                "tokens": tokens,
                "ner_tags": labels
            })

    return combined_ner_data


input_files = ['test20.json', 'pl1.json', 'test.json', 'ks4.json', 'ks2.json', 'ks3.json']

combined_ner_data = convert_to_ner_format(input_files)

output_file = 'upd_NER.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(combined_ner_data, f, ensure_ascii=False, indent=4)

print(f"data saved to {output_file}")

