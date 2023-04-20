import sqlite3
import unicodedata

class DB():

    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()

    def insert_person(self, name):
        name = unicodedata.normalize('NFD', name)
        self.c.execute('SELECT * FROM persons WHERE name=?', (name,))
        exists = self.c.fetchone()

        self.c.execute("SELECT MAX(uid) FROM persons")
        max_uid = self.c.fetchone()[0]

        if not exists:
            self.c.execute('INSERT INTO persons (name, uid) VALUES (?, ?)', (name, max_uid+1))
            self.conn.commit()
            return max_uid+1
        else:
            return -1

    def show_people(self):
        self.c.execute("SELECT name FROM persons")
        results = self.c.fetchall()
        people = [result[0] for result in results]
        print("People in the database:")
        print(people)

    def check_exists(self, name):
        name = unicodedata.normalize('NFD', name)
        self.c.execute("SELECT name FROM persons WHERE name=?", (name,))
        result = self.c.fetchone()
        if not result:
            print(f"{name} is not in the database.")
        else:
            print(f"{name} is already in the database.")
