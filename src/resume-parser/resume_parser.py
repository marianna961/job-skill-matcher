import re
import docx
import pdfplumber
import sqlite3

def extract_info_from_txt(text):
    data = {
        "specialization": None,
        "desired_position": None,
        "previous_positions": [],
        "tasks_at_previous_jobs": [],
        "education": None,
        "skills": [],
        "about_me": None
    }

    desired_position_match = re.search(r"Желаемая должность и зарплата\s*\n([^\n]+)", text)
    if desired_position_match:
        data["desired_position"] = desired_position_match.group(1).strip()

    specialization_match = re.search(r"Специализации:\s*\n([^\n]+)", text)
    if specialization_match:
        data["specialization"] = specialization_match.group(1).strip()

    experience_blocks = re.findall(r"([^-\n]+)\n[^\n]+\n([^—\n]+)[^\n]*—[^\n]+\n([\s\S]+?)(?=\n\n|\Z)", text)
    for block in experience_blocks:
        company = block[0].strip()
        position = block[1].strip()
        tasks = [task.strip("- ") for task in block[2].split("\n") if task.strip()]
        data["previous_positions"].append({"company": company, "position": position})
        data["tasks_at_previous_jobs"].append({"company": company, "tasks": tasks})

    education_match = re.search(r"Образование\s*\n([\s\S]+?)\n\n", text)
    if education_match:
        data["education"] = education_match.group(1).strip()

    skills_match = re.search(r"Навыки\s*\n([\s\S]+?)(?=\n\n|\Z)", text)
    if skills_match:
        skills_text = skills_match.group(1).strip()
        data["skills"] = [skill.strip() for skill in skills_text.split(";") if skill.strip()]

    about_me_match = re.search(r"Обо мне\s*\n([\s\S]+?)(?=\n\n|\Z)", text)
    if about_me_match:
        data["about_me"] = about_me_match.group(1).strip()

    return data


def parse_doc(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Ошибка при парсинге DOC: {e}")
        return None


def parse_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages])
        return text
    except Exception as e:
        print(f"Ошибка при парсинге PDF: {e}")
        return None


def parse_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Ошибка при чтении TXT: {e}")
        return None


def save_data_to_db(data):
    connection = sqlite3.connect("resumes.db")
    cursor = connection.cursor()
    try:

        cursor.execute("""
        INSERT INTO users (name, email, phone, location)
        VALUES (?, ?, ?, ?)
        """, ("Фамилия Имя", "example@example.com", "+7 (999) 123-45-67", "Москва"))
        user_id = cursor.lastrowid

        cursor.execute("""
        INSERT INTO resumes (user_id, desired_position, specialization, education, about_me)
        VALUES (?, ?, ?, ?, ?)
        """, (user_id, data["desired_position"], data["specialization"], data["education"], data["about_me"]))
        resume_id = cursor.lastrowid
        
        for skill in data["skills"]:
            cursor.execute("""
            INSERT INTO skills (resume_id, skill_name)
            VALUES (?, ?)
            """, (resume_id, skill))

        for job in data["previous_positions"]:
            company = job["company"]
            position = job["position"]
            tasks = "; ".join(data["tasks_at_previous_jobs"][data["previous_positions"].index(job)]["tasks"])
            cursor.execute("""
            INSERT INTO experience (resume_id, company, position, tasks)
            VALUES (?, ?, ?, ?)
            """, (resume_id, company, position, tasks))      

        connection.commit()  

    except Exception as e:
        print(f"Ошибка при сохранении данных: {e}")
        connection.rollback()

    finally:
        connection.close()

def parse_resume(file_path, format_type):
    if format_type == 'doc':
        text = parse_doc(file_path)
    elif format_type == 'pdf':
        text = parse_pdf(file_path)
    elif format_type == 'txt':
        text = parse_txt(file_path)
    else:
        print("Неизвестный формат данных.")
        return None

    if text:
        return extract_info_from_txt(text)
    return None