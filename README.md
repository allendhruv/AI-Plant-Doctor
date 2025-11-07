<h2>🌿 AI Plant Doctor</h2>

AI Plant Doctor is a plant leaf disease detection application.  
You upload a leaf image, the app detects the leaf, identifies the disease, and shows the recommended treatment.

✨ Features

- Detects and classifies plant leaf diseases from images
- Shows disease name and solution steps
- Web-based interface built with Flask
- Uses a trained deep learning model (`plant_disease_model.h5`)

🛠 Requirements

- **Python Version:** 3.10 or 3.11  
  (The model and dependencies may not work correctly on Python 3.12+)
- Install required libraries first.

🚀 How to Run the Application:
Install Dependencies:
bash - pip install -r req.txt

Run the application: 
python app.py

Open the link shown in your terminal, usually:
http://127.0.0.1:5000/



<h3>📂 Project Structure</h3>
AI Plant Doctor/
│
│
├── .env                        # containing openrouter api
├── app.py                      # Main Flask application
├── predict.py                  # Disease detection and prediction code
├── model/
│   ├── plant_disease_model.h5  # Trained model file
│   └── classes.txt             # Plant disease label list
├── static/
│   └── images/                 # Static images
└── templates/
    ├── home.html
    ├── contact.html
    ├── index.html
    ├── login.html
    ├── register.html
    ├── recent.html
    └── layout.html


<h3>🔧 Model Details</h3>

Model Type: Convolutional Neural Network
Input: RGB Leaf Image
Output: Predicted plant disease label + recommended solution


<h3>📄 License</h3>

This project is provided for personal, educational, and research use.
For commercial usage, please contact the author.





