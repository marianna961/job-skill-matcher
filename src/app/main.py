import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer, util
from ner_pipeline import extract_entities
import pandas as pd
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

university_tier_list = {
    "ИТМО": 1,
    "МФТИ": 1,
    "Московский физико-технический институт": 1,
    "ВШЭ": 1,
    "Ломоносова": 1,
    "Высшая школа экономики": 1,
    "СПБГУ": 1,
    "Санкт-Петербургский государственный университет": 1,
    "МГУ": 1,
    "Московский государственный университет": 1,
    "МГТУ им. Н. Э. Баумана": 2,
    "МГТУ": 2,
    "Баумана": 2,
    "УРФУ": 2,
    "Уральский федеральный университет": 1,
    "Иннополис": 2,
    "Ельцина": 2,
    "Санкт-Петербургский политехнический университет Петра Великого": 1,
    "МИСИС": 2,
    "МИФИ": 2,
    "МИРЭА": 2,
    "Южный федеральный университет": 2,
    "ЮФУ": 2,
    "Финансовый университет при правительстве Россий Федерации": 2,
    "Финансовый университет": 2,
    "Финансовый при правительстве": 2,
    "КФУ": 2,
    "НГУ": 2,
    "Российский технологический университет": 2,
    "МАИ": 2,
    "Московский авиационный": 2

}

expert_keywords = ["эксперт", "глубокое знание", "профессионал", "экспертный уровень", "специалист"]
advanced_keywords = ["продвинутый", "опыт работы", "уверенное владение"]
basic_keywords = ["базовый", "основы", "начальный"]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_university_score(education_str):
    for uni, tier in university_tier_list.items():
        if uni.lower() in education_str.lower():
            return uni, tier
    return "Неизвестно", 3


def get_candidate_level(skill, profession_skills):
    skill_lower = skill.lower()
    text_lower = profession_skills.lower()

    matches = [word for word in profession_skills.split() if skill_lower in word.lower()]
    if not matches:
        return 0, ""

    matched_word = matches[0]

    tokens = text_lower.split()
    for i, token in enumerate(tokens):
        if skill_lower in token:
            # Только слово до скилла
            if i > 0:
                previous_word = tokens[i - 2]
                if any(word in previous_word for word in expert_keywords):
                    return 3, matched_word
                elif any(word in previous_word for word in advanced_keywords):
                    return 2, matched_word
                elif any(word in previous_word for word in basic_keywords):
                    return 1, matched_word
                else:
                    return 1, matched_word
            else:
                # Если скилл — первое слово в тексте, до него ничего нет
                return 1, matched_word

    return 1, matched_word



def compare_with_matrix(profession_skills, selected_job):
    df = pd.read_csv("matrix.csv", sep=';')
    results = []
    ideal_vector = []
    candidate_vector = []

    for _, row in df.iterrows():
        skill = row['Skill']
        required_level = row.get(selected_job, np.nan)
        if pd.isna(required_level):
            continue
        
        candidate_level, matched_word = get_candidate_level(skill, profession_skills)

        ideal_vector.append(int(required_level))
        candidate_vector.append(candidate_level)

        results.append({
            'skill': skill,
            'matched_resume_skill': matched_word,
            'required_level': int(required_level),
            'candidate_level': candidate_level,
            'match_percentage': min(100, round((candidate_level / int(required_level)) * 100, 2)) if candidate_level else 0
        })

    similarity = util.cos_sim(np.array([candidate_vector], dtype=np.float32),
                              np.array([ideal_vector], dtype=np.float32)).item()

    return results, similarity * 100


@app.route('/', methods=['GET', 'POST'])
def upload_resume():
    if request.method == 'POST':
        if 'resume' not in request.files:
            return "No file part"
        file = request.files['resume']
        if file.filename == '':
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            resume_text = read_file(filepath)

            raw_results = extract_entities(resume_text)

            # нужно удалять ##
            df = pd.DataFrame([raw_results])
            df = df.applymap(lambda x: (
                x.replace(" ##", "")
                .replace("##", "")
                .replace("  ", " ")
                .replace(" @ ", "@")
                .replace(" . ", ".")
                .strip()
            ) if isinstance(x, str) else x)

            cleaned = df.iloc[0].to_dict()

            selected_job = request.form['job_role']

            profession_skills = cleaned.get('PROFESSION_SKILLS', '')
            education = cleaned.get('EDUCATION', '')
            work_experience = cleaned.get('WORK_EXP', '')

            comparison_result, similarity_score = compare_with_matrix(profession_skills, selected_job)
            candidate_skills = [r['matched_resume_skill'] for r in comparison_result if r['matched_resume_skill']]

            university, university_score = get_university_score(education)

            matrix = {
                'comparison_result': comparison_result,
                'candidate_skills': candidate_skills,
                'university': university,
                'university_score': university_score,
                'similarity_score': round(similarity_score, 2)
            }
            


            return render_template('index.html', cleaned=cleaned, matrix=matrix, selected_job=selected_job, work_exp=work_experience)
                                   
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
