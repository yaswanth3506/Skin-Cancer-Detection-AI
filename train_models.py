import pandas as pd
import numpy as np
import os
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    RandomFlip,       # Added for Fix B
    RandomRotation,   # Added for Fix B
    RandomZoom        # Added for Fix B
)
from tensorflow.keras.utils import to_categorical

# -------------------------------
# DATASET PATHS
# -------------------------------
csv_path = 'datasets/HAM10000_cancerdata.csv'
image_folder_1 = 'datasets/HAM10000_images_part_1'
image_folder_2 = 'datasets/HAM10000_images_part_2'

# -------------------------------
# LOAD CSV
# -------------------------------
df = pd.read_csv(csv_path)
print(df.head())

# -------------------------------
# LABEL ENCODING
# -------------------------------
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['dx'])
classes = label_encoder.classes_
print("Initial HAM10000 Classes:", classes)

# -------------------------------
# LOAD IMAGES (HAM10000 + NORMAL SKIN)
# -------------------------------
images = []
labels = []
IMG_SIZE = 64

# --- A. Load HAM10000 Images ---
for index, row in df.iterrows():
    image_id = row['image_id']
    label = row['label']

    image_path1 = os.path.join(image_folder_1, image_id + '.jpg')
    image_path2 = os.path.join(image_folder_2, image_id + '.jpg')
    image_path = None

    if os.path.exists(image_path1):
        image_path = image_path1
    elif os.path.exists(image_path2):
        image_path = image_path2

    if image_path:
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img).astype('float32') / 255.0
            images.append(img)
            labels.append(label)
        except Exception as e:
            print("Error loading HAM image:", image_path)

# --- B. Load Normal Skin Images ---
normal_skin_folder = 'datasets/NORMAL_SKIN'
normal_skin_label_idx = len(classes) 

if os.path.exists(normal_skin_folder):
    print(f"Loading normal skin samples. Mapping to label index: {normal_skin_label_idx}")
    for file_name in os.listdir(normal_skin_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(normal_skin_folder, file_name)
            try:
                img = Image.open(image_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                img = np.array(img).astype('float32') / 255.0
                images.append(img)
                labels.append(normal_skin_label_idx)
            except Exception as e:
                print("Error loading normal skin image:", image_path)

    classes = np.append(classes, 'normal_skin')
else:
    print("⚠️ Warning: datasets/NORMAL_SKIN folder was not found!")

# -------------------------------
# CONVERT TO NUMPY
# -------------------------------
X = np.array(images)
y = np.array(labels)

print("Total Images (HAM10000 + Normal Skin):", len(X))
print("Final Classes List:", classes)

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

class_weights = dict(enumerate(class_weights))

print("Class Weights:")
print(class_weights)

# -------------------------------
# ONE HOT ENCODING
# -------------------------------
y_train = to_categorical(y_train, num_classes=len(classes))
y_test = to_categorical(y_test, num_classes=len(classes))

# -------------------------------
# BUILD CNN MODEL (WITH AUGMENTATION)
# -------------------------------
model = Sequential()

# Fix B: Dynamic Data Augmentation Layers
model.add(RandomFlip("horizontal_and_vertical", input_shape=(64, 64, 3)))
model.add(RandomRotation(0.2))
model.add(RandomZoom(0.2))

# Core Feature Extraction Layers
model.add(Conv2D(32, (3, 3), activation='relu'))  # input_shape is handled by RandomFlip above
model.add(MaxPool2D(2, 2))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPool2D(2, 2))

# Classification Head
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(classes), activation='softmax'))

# -------------------------------
# COMPILE AND TRAIN
# -------------------------------
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Consider increasing epochs to 15 or 20 later since augmentation makes learning slower but more accurate
history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32
)

# -------------------------------
# SAVE OUTPUTS
# -------------------------------
model.save('models/skin_cancer_model.h5')
np.save('models/class_names.npy', classes)
print("Model and Class labels saved successfully with Augmentation Layers.")