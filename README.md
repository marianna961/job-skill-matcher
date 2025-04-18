# Job-Skill-Matcher

![img](image/result.jpg)

## Описание проекта

Job-Skill-Matcher — это веб-приложение, которое анализирует резюме кандидатов и сравнивает их навыки с требованиями для определенной профессии. Проект использует NLP (Natural Language Processing) для извлечения ключевых данных из резюме, таких как навыки, образование и опыт работы, а затем вычисляет соответствие кандидата требованиям на основе матрицы компетенций.

Основные функции:

- Извлечение навыков из резюме с помощью NER (Named Entity Recognition).
- Сравнение навыков кандидата с матрицей компетенций.
- Оценка уровня владения навыками (0–3) на основе текста резюме.
- Визуализация результатов в виде таблицы с процентами совпадения.
- Учет университета и его рейтинга при расчете общей оценки.

---

## Структура проекта

```
JOB-SKILL-MATCHER/
├── app/
│   └── main.py             # Основной файл Flask-приложения
│   └── templates/
│   	└── index.html      # HTML-шаблон для отображения результатов
│   └── matrix.csv          # Матрица компетенций
│   └── ner_pipeline.py     # Пайплайн для извлечения данных из резюме
├── ner_parser/
│   └── ner_model/          # NER-модель для извлечения навыков
├── requirements.txt        # Зависимости Python
└── README.md               # Документация проекта
```

---

## Установка

1. **Клонируйте репозиторий:**

   ```bash
   git clone https://github.com/marianna961/job-skill-matcher.git
   cd job-skill-matcher
   ```
2. **Создайте виртуальное окружение:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Для Linux/MacOS
   venv\Scripts\activate     # Для Windows
   ```
3. **Установите зависимости:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Подготовьте данные:**

   - Скачайте модель по ссылке [kaggle](https://www.kaggle.com/models/mariannach/ner_job_skill_matcher/) и расспакуйте в папке ner_parser.
   - Убедитесь, что файл `matrix.csv` находится в папке app.
   - Файл должен содержать список навыков и требуемых уровней для каждой профессии.
5. **Запустите приложение:**

   ```bash
   cd src/app
   python main.py
   ```
6. **Откройте браузер:**
   Перейдите по адресу [http://127.0.0.1:5000](http://127.0.0.1:5000), чтобы использовать приложение.

---

## Использование

1. **Загрузите резюме:**

   - Выберите файл резюме (в папке test_resume_datasets лежат примеры txt файлов с резюме для анализа).
   - Выберите вакансию из выпадающего списка.
2. **Просмотрите результаты:**

   - Приложение покажет распознанные данные (личная информация, навыки, образование).
   - Будет показана таблица с оценкой каждого навыка и общая оценка кандидата.

---

## Конфигурация

### `matrix.csv`

Файл содержит матрицу компетенций. Структура файла:

```csv
Skill,DATA_SCIENTIST,DATA_ENGINEER,TECHNICAL_ANALYST_IN_AI,MANAGER_IN_AI
Python,3,2,2,1
SQL,2,3,2,1
...,3,1,2,0
```

- **Skill:** Название навыка.
- **DATA_SCIENTIST, DATA_ENGINEER, ...:** Требуемые уровни для каждой профессии.

---

## Результаты оценки NER модели

| Параметр                  | Значение    |
| --------------------------------- | ------------------- |
| **eval_loss**               | 0.0741              |
| **eval_precision**          | 0.9522              |
| **eval_recall**             | 0.9541              |
| **eval_f1**                 | 0.9531              |
| **eval_accuracy**           | 0.9792              |
| **eval_runtime**            | 21.07 секунд |
| **eval_samples_per_second** | 5.124               |
| **eval_steps_per_second**   | 0.332               |
| **epoch**                   | 3.0                 |

## Авторы проекта:

|                                          | git            |
| ---------------------------------------- | -------------- |
| Черткова Марианна (Lead) | marianna961    |
| Симонова Ксения            | kseniasimonova |
| Шалимова Полина            | polina-cmd     |
