import sqlite3

def init_db():
    conn = sqlite3.connect('plant_predictions.db')
    c = conn.cursor()

    # Users table (if not already created)
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')

    # Predictions table with user_id
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            prediction TEXT,
            solution TEXT,
            timestamp TEXT,
            user_id INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()

def register_user(username, password):
    conn = sqlite3.connect('plant_predictions.db')
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username already exists
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('plant_predictions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    result = c.fetchone()
    conn.close()
    return result is not None

def get_user_id(username):
    conn = sqlite3.connect('plant_predictions.db')
    c = conn.cursor()
    c.execute('SELECT id FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def insert_prediction(image_path, prediction, solution, user_id):
    conn = sqlite3.connect('plant_predictions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (image_path, prediction, solution, timestamp, user_id)
        VALUES (?, ?, ?, datetime('now'), ?)
    ''', (image_path, prediction, solution, user_id))
    conn.commit()
    conn.close()

def get_recent_predictions(user_id):
    conn = sqlite3.connect('plant_predictions.db')
    c = conn.cursor()
    c.execute('''
        SELECT image_path, prediction, solution, timestamp
        FROM predictions
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (user_id,))
    recent = c.fetchall()
    conn.close()
    return recent

def clear_predictions(user_id):
    conn = sqlite3.connect('plant_predictions.db')
    c = conn.cursor()
    c.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()
