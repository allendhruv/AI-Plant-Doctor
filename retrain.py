import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

# -------------------------------
# CONFIGURATION
# -------------------------------

DATASET_DIR = "DatasetRoot"
MODEL_SAVE_PATH = "model/plant_disease_model.h5"
CLASSES_TXT_PATH = "model/classes.txt"
IMG_SIZE = (224, 224)  # MobileNetV2 requires at least 96x96; 224 gives better results
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4

os.makedirs("model", exist_ok=True)

# -------------------------------
# DATA GENERATORS (with augmentation)
# -------------------------------

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    shear_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Save class labels
class_labels = list(train_gen.class_indices.keys())
with open(CLASSES_TXT_PATH, "w") as f:
    f.writelines("\n".join(class_labels))
print(f"[INFO] Found {len(class_labels)} classes.")
print(f"[INFO] Saved class names to {CLASSES_TXT_PATH}")

# -------------------------------
# MODEL ARCHITECTURE (Transfer Learning)
# -------------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
base_model.trainable = False  # freeze base for initial training

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(class_labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# TRAINING
# -------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

print(f"[INFO] Training completed. Model saved to {MODEL_SAVE_PATH}")

# -------------------------------
# (Optional) Unfreeze some base layers for fine-tuning
# -------------------------------
# After initial training, you can fine-tune a few deeper layers:
"""
base_model.trainable = True
for layer in base_model.layers[:-30]:  # freeze all but last 30 layers
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # smaller LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[early_stop, checkpoint]
)
print("[INFO] Fine-tuning completed.")
"""
