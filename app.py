from flask import Flask, render_template, request, session, redirect, url_for, flash
from database import init_db, insert_prediction, get_recent_predictions, register_user, verify_user, clear_predictions, get_user_id
from werkzeug.utils import secure_filename
from ai_helper import get_disease_solution
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os
import re
from PIL import Image    # <-- added this

app = Flask(__name__)
app.secret_key = 'supersecretkey123'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Allowed image types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def is_allowed_file(file):
    if "." not in file.filename:
        return False
    ext = file.filename.rsplit(".", 1)[1].lower()

    # Check extension
    if ext not in ALLOWED_EXTENSIONS:
        return False

    # Check MIME type
    if not file.mimetype.startswith("image/"):
        return False

    return True

# Load model and labels
model = load_model("model/plant_disease_model.h5")
with open("model/classes.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

def detect_leaf_presence(filepath):
    img = cv2.imread(filepath)
    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

    return green_ratio > 0.05

@app.route('/')
def home():
    return redirect(url_for('home_page'))

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if register_user(username, password):
            return redirect(url_for('login'))
        else:
            return render_template('register.html', error="Username already exists")
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if verify_user(username, password):
            session['username'] = username
            session['user_id'] = get_user_id(username)
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file.')

    # *** IMAGE VALIDATION STARTS HERE ***
    if not is_allowed_file(file):
        return render_template('index.html', error='Only PNG, JPG, JPEG, and WEBP image files are allowed.')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    # Verify actual image content
    try:
        Image.open(filepath).verify()
    except:
        os.remove(filepath)
        return render_template('index.html', error='Invalid or corrupted image.')
    # *** VALIDATION ENDS HERE ***

    leaf_present = detect_leaf_presence(filepath)
    if not leaf_present:
        return render_template('index.html', prediction="No leaf detected.", image_path=filepath, solution=[])

    try:
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)].strip()
        print(f"[INFO] Predicted class: {predicted_class}")

    except Exception as e:
        return render_template('index.html', error=f'Error processing image: {e}')

    if "healthy" in predicted_class.lower():
        raw_solution = "No disease detected. The leaf looks healthy! 😊 Keep monitoring your plant regularly and maintain good practices."
        solution_points = [raw_solution]
    else:
        try:
            raw_solution = get_disease_solution(predicted_class)
            solution_points = re.split(r'\d+\.\s*', raw_solution)[1:] or [raw_solution]
        except Exception as e:
            raw_solution = f"Error fetching AI solution: {str(e)}"
            solution_points = [raw_solution]

    user_id = session.get("user_id")
    if user_id:
        insert_prediction(filepath, predicted_class, raw_solution, user_id)

    return render_template('index.html', prediction=predicted_class, solution=solution_points, image_path=filepath)

@app.route('/recent')
def recent():
    user_id = session.get("user_id")
    if not user_id:
        flash("Please log in to view recent predictions.")
        return redirect(url_for("login"))

    recent_data = get_recent_predictions(user_id)
    return render_template('recent.html', recent=recent_data)

@app.route('/clear_predictions', methods=['POST'])
def clear_predictions_route():
    user_id = session.get("user_id")
    if not user_id:
        flash("You must be logged in to clear predictions.")
        return redirect(url_for("login"))

    clear_predictions(user_id)
    return redirect(url_for('recent'))

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    init_db()
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
