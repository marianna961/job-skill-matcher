import json
from datasets import Dataset
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from collections import Counter

with open('datasets_labeled_converted/upd_NER.json', 'r', encoding='utf-8') as f:
    ner_data = json.load(f)

dataset = Dataset.from_dict({
    "tokens": [item["tokens"] for item in ner_data],
    "ner_tags": [item["ner_tags"] for item in ner_data]
})

# Подсчет частоты меток
all_tags = [tag for tags in dataset["ner_tags"] for tag in tags]
tag_counts = Counter(all_tags)
print("Частота меток:", tag_counts)

# Создаем словарь меток с фиксированным порядком
unique_labels = sorted(set(tag for tags in dataset["ner_tags"] for tag in tags))
label_to_id = {label: i for i, label in enumerate(unique_labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

print("Label to ID:", label_to_id)

# https://huggingface.co/google-bert/bert-base-multilingual-cased
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Функция для токенизации и выравнивания меток
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True, 
        padding="max_length", 
        max_length=128
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(label_to_id[label[word_idx]])
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Разделяем данные на обучающую и тестовую выборки
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

# Загрузка предобученной модели
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-multilingual-cased", 
    num_labels=len(unique_labels), 
    id2label=id_to_label, 
    label2id=label_to_id
)

# Настройки обучения
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs", #tensorboard --logdir=./logs
    logging_steps=100
)

# Загружаем метрику seqeval
metric = evaluate.load("seqeval")

# Функция для расчета метрик
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Создаем Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# Оценка модели
metrics = trainer.evaluate()
print(metrics)

with open("evaluation_results.txt", "w", encoding="utf-8") as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value}\n")

model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")

print("DONE")