import sqlite3

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
        print("Данные успешно сохранены в базу данных!")

    except Exception as e:
        print(f"Ошибка при сохранении данных: {e}")
        connection.rollback()

    finally:
        connection.close()