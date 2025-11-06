import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
import tensorflow as tf
import os

# Load the trained model
model = load_model("model/plant_disease_model.h5")

# Print model summary (CNN structure)
print("🧠 Model Summary:")
model.summary()

# Load class names
with open("model/classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
num_classes = len(class_names)

# Function to preprocess and predict a single image
def predict_image(img_path):
    img_size = (128, 128)  # Must match training size
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    
    print("📊 Image Matrix (shape):", img_array.shape)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    
    print(f"✅ Prediction: {predicted_class}")
    return predicted_class

# OPTIONAL: Evaluate model performance if validation data is available
def evaluate_model():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Assuming you have a validation folder structure like: validation/class_name/images...
    validation_dir = "dataset"  # update if needed

    if not os.path.exists(validation_dir):
        print("⚠️ Skipping evaluation: No validation data found.")
        return

    datagen = ImageDataGenerator(rescale=1.0/255)
    val_gen = datagen.flow_from_directory(
        validation_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Predict and calculate classification metrics
    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes

    print("📈 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

# Run prediction
predict_image("test_images/download.webp")

# OPTIONAL: Evaluate performance (if validation data is available)
# evaluate_model()
# Example for checking accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up the test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'DatasetRoot/',  # folder containing test images grouped in subfolders by class
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"✅ Accuracy: {accuracy * 100:.2f}%")
