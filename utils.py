import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -------------------------------------------------------------------------
# GLOBAL MODEL INITIALIZATION
# -------------------------------------------------------------------------
MODEL_PATH = 'models/skin_cancer_model.h5'
LABELS_PATH = 'models/class_names.npy'

model = None
class_names = {}

if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    try:
        model = load_model(MODEL_PATH)
        classes = np.load(LABELS_PATH, allow_pickle=True)
        class_names = {i: label for i, label in enumerate(classes)}
        print("[-] Inference Engine Loaded Successfully. Classes mapped:", class_names)
    except Exception as e:
        print(f"[!] Critical Error: Could not initialize model weights: {e}")
else:
    print("[!] Warning: Model files not found. Run train_models.py first.")


# -------------------------------------------------------------------------
# DEFENSIVE COMPUTER VISION GATEWAY (SKIN SCANNER)
# -------------------------------------------------------------------------
def is_skin_image(img_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
            
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        skin_mask = cv2.inRange(hsv_img, lower_skin, upper_skin)
        total_pixels = img.shape[0] * img.shape[1]
        skin_pixels = cv2.countNonZero(skin_mask)
        
        skin_percentage = (skin_pixels / total_pixels) * 100
        return skin_percentage >= 15.0
    except Exception:
        return False


# -------------------------------------------------------------------------
# MODEL PREDICTION PIPELINE (ORIGINAL MATH)
# -------------------------------------------------------------------------
def predict_image(img_path):
    """
    Runs original model predictions and extracts both the 1st and 2nd highest choices.
    """
    if model is None:
        return "Model uninitialized", 0.0, "N/A", 0.0

    try:
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)[0]
        
        # Sort indices by probability in descending order
        sorted_indices = np.argsort(preds)[::-1]
        
        # Top Prediction
        top_idx = int(sorted_indices[0])
        top_prob = float(preds[top_idx])
        top_label = class_names.get(top_idx, "unknown")
        
        # Second Top Prediction
        sec_idx = int(sorted_indices[1])
        sec_prob = float(preds[sec_idx])
        sec_label = class_names.get(sec_idx, "unknown")

        return top_label, top_prob * 100, sec_label, sec_prob * 100

    except Exception as e:
        print(f"Error reading image file {img_path}: {e}")
        return "Corrupted/Invalid Image File", 0.0, "N/A", 0.0


def get_true_label(filename):
    return "unknown"