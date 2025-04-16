import os
import pandas as pd
from flask import Flask, request, render_template
from ner_pipeline import extract_entities
# from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from find_match import find_best_match

# Загрузка матрицы компетенций
if not os.path.exists('matrix.csv'):
    raise FileNotFoundError("Файл matrix.csv не найден")
competency_df = pd.read_csv('matrix.csv', delimiter=";")
competency_df.set_index('Skill', inplace=True)

# Список университетов
university_tier_list = {
    "ИТМО": 1,
    "МФТИ": 1,
    "Московский физико-технический институт": 1,
    "ВШЭ": 1,
    "Высшая школа экономики": 1,
    "СПБГУ": 1,
    "Санкт-Петербургский государственный университет": 1,
    "МГУ": 1,
    "Московский государственный университет": 1,
    "Ломоносова": 1,
    "МГТУ им. Н. Э. Баумана": 2,
    "МГТУ": 2,
    "Баумана": 2,
    "УРФУ": 2,
    "Уральский федеральный университет": 2,
    "Иннополис": 2,
    "Петра Великого": 2,
    "Санкт-Петербургский политехнический университет": 2,
    "МИСИС": 2,
    "МИФИ": 2
}

# Модель эмбеддингов
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

app = Flask(__name__)

def evaluate_university(university_name):
    return university_tier_list.get(university_name, 3)

def determine_skill_level(resume_text, skill, base_level):
    lower_resume = resume_text.lower()
    skill_lower = skill.lower()
    
    expert_keywords = ["эксперт", "глубокое знание", "профессионал", "экспертный уровень"]
    advanced_keywords = ["продвинутый", "опыт работы", "уверенное владение"]
    basic_keywords = ["базовый", "основы", "начальный"]
    
    context = lower_resume.split(skill_lower)[-1] if skill_lower in lower_resume else ""
    
    for keyword in expert_keywords:
        if keyword in context:
            return 3
    for keyword in advanced_keywords:
        if keyword in context:
            return 2
    for keyword in basic_keywords:
        if keyword in context:
            return 1
    
    return 0

@app.route("/", methods=["GET", "POST"])
def upload_resume():
    entities_list = []
    cleaned_results = None
    competency_matrix = None
    selected_job = None
    skill_matrix_html = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("resume")
        selected_job = request.form.get("job_role")

        if uploaded_file and selected_job:
            try:
                resume_text = uploaded_file.read().decode("utf-8")
                results = extract_entities(resume_text)
                

                for entity_group, text in results.items():
                    entities_list.append({
                        "entity_group": entity_group,
                        "text": text
                    })

                df = pd.DataFrame([results])
                df = df.applymap(lambda x: (
                    x.replace(" ##", "")
                     .replace("##", "")
                     .replace("  ", " ")
                     .replace(" @ ", "@")
                     .replace(" . ", ".")
                     .strip()
                ) if isinstance(x, str) else x)

                cleaned_results = df.to_dict(orient="records")[0]

                skills_list = results.get("skills", [])
                if isinstance(skills_list, str):
                    skills_list = skills_list.split()

                if not skills_list:
                    print("В резюме не найдены навыки")
                


                university = results.get("university", "")
                matrix_skills = competency_df.index.tolist()
                comparison_result = []

                for matrix_skill in matrix_skills:
                    best_resume_skill = find_best_match(matrix_skill, skills_list, embedding_model)
                    required_level = competency_df.loc[matrix_skill, selected_job]
                    
                    if best_resume_skill:
                        candidate_level = determine_skill_level(resume_text, best_resume_skill, required_level)
                    else:
                        candidate_level = 0
                    
                    match_percentage = (candidate_level / required_level * 100) if required_level > 0 else 0
                    
                    comparison_result.append({
                        'skill_matrix': matrix_skill,
                        'matched_resume_skill': best_resume_skill or "Не найдено",
                        'required_level': required_level,
                        'candidate_level': candidate_level,
                        'match_percentage': f"{match_percentage:.2f}%"
                    })

                candidate_vector = [item['candidate_level'] for item in comparison_result]
                ideal_vector = [item['required_level'] for item in comparison_result]
                
                skill_similarity = cosine_similarity([candidate_vector], [ideal_vector])[0][0]
                university_score = evaluate_university(university)
                
                total_score = 0.8 * skill_similarity + 0.2 * (university_score / 3)

                skill_matrix_df = pd.DataFrame(comparison_result)
                skill_matrix_html = skill_matrix_df.to_html(index=False, classes="table table-bordered")

                
                competency_matrix = {
                    "university": university,
                    "university_score": university_score,
                    "skill_similarity": skill_similarity,
                    "total_score": total_score
                }
            except Exception as e:
                error = f"Ошибка обработки резюме: {str(e)}"

    return render_template(
        "index.html",
        entities=entities_list,
        cleaned=cleaned_results,
        matrix=competency_matrix,
        selected_job=selected_job,
        skill_matrix_html=skill_matrix_html,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)