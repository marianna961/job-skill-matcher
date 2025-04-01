import sqlite3

def read_data_from_db():
    connection = sqlite3.connect("resumes.db")
    cursor = connection.cursor()

    print("Таблица users:")
    cursor.execute("SELECT * FROM users")
    for row in cursor.fetchall():
        print(row)

    print("\nТаблица resumes:")
    cursor.execute("SELECT * FROM resumes")
    for row in cursor.fetchall():
        print(row)

    print("\nТаблица skills:")
    cursor.execute("SELECT * FROM skills")
    for row in cursor.fetchall():
        print(row)

    print("\nТаблица experience:")
    cursor.execute("SELECT * FROM experience")
    for row in cursor.fetchall():
        print(row)

    connection.close()

if __name__ == "__main__":
    read_data_from_db()