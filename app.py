from flask import Flask, render_template, request, session, redirect, url_for, flash
from database import init_db, insert_prediction, get_recent_predictions, register_user, verify_user, clear_predictions, get_user_id
from werkzeug.utils import secure_filename
from ai_helper import get_disease_solution
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import re

app = Flask(__name__)
app.secret_key = 'supersecretkey123'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

model = load_model("model/plant_disease_model.h5")

# Load class labels
with open("model/classes.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# List of keywords to determine if it's a valid leaf disease
leaf_keywords = [cls.lower() for cls in class_labels if any(word in cls.lower() for word in ["leaf", "blight", "spot", "mold", "healthy"])]


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
            session['user_id'] = get_user_id(username)  # Save user_id to session
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_id', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    return redirect(url_for('home_page'))  # / → /home

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print("📥 predict route called")
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded.')

    file = request.files['file']
    print(f"Uploaded file: {file.filename}")
    if file.filename == '':
        return render_template('index.html', error='No selected file.')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    try:
        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Model prediction
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)].strip()
        print(f"Predicted class: {predicted_class}")
    except Exception as e:
        return render_template('index.html', error=f'Error processing image: {e}')

    # Check if the predicted class is a known plant-related label
    if predicted_class not in class_labels:
        return render_template('index.html', prediction="No leaf detected.", image_path=filepath, solution=[])

    # Check if it's a valid leaf issue (blight, spot, mold, etc.)
    if not any(keyword in predicted_class.lower() for keyword in leaf_keywords):
        return render_template('index.html', prediction="No leaf detected.", image_path=filepath, solution=[])

    # Handle healthy leaf
    if "healthy" in predicted_class.lower():
        raw_solution = "No disease detected. The leaf looks healthy! 😊\nKeep monitoring your plant regularly and maintain good agricultural practices."
        solution_points = [raw_solution]
    else:
        try:
            raw_solution = get_disease_solution(predicted_class)
            solution_points = re.split(r'\d+\.\s*', raw_solution)[1:]
        except Exception as e:
            raw_solution = f"Error fetching AI solution: {str(e)}"
            solution_points = [raw_solution]

    # Store result in DB if user logged in
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

    recent = get_recent_predictions(user_id)
    return render_template('recent.html', recent=recent)

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
