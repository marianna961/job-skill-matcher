import sqlite3

def save_data_to_db(data):
    connection = sqlite3.connect("resumes.db")
    cursor = connection.cursor()

    try:
        cursor.execute("""
        INSERT INTO users (name, email, phone, location)
        VALUES (?, ?, ?, ?)
        """, (data["name"], data["email"], data["phone"], data["location"]))
        user_id = cursor.lastrowid

        cursor.execute("""
        INSERT INTO resumes (
            user_id, desired_position, specialization, education, skills, experience, about_me
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            data["desired_position"],
            data["specialization"],
            data["education"],
            data["skills"],
            "; ".join(data["experience"]),
            data["about_me"]
        ))

        connection.commit()
        print("Данные успешно сохранены в базу данных!")
        return user_id

    except Exception as e:
        print(f"Ошибка при сохранении данных: {e}")
        connection.rollback()

    finally:
        connection.close()