import os
import numpy as np
import cv2
import base64
import traceback
import sys
import urllib.request
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "simple_secret_key"
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # Avoid large upload crashes

# Absolute paths for Linux compatibility
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
model = None

# Hardcoded class mapping
class_names = np.array([
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic nevi',
    'Vascular lesions',
    'normal_skin'
])

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)



def get_model():
    global model, class_names

    if model is not None:
        return model, class_names

    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(current_dir, 'models', 'skin_cancer_model.h5')
    CLASS_PATH = os.path.join(current_dir, 'models', 'class_names.npy')

    if not os.path.exists(MODEL_PATH):
        print("--> ERROR: Model file not found.")
        return None, class_names

    try:
        model = load_model(MODEL_PATH)

        if os.path.exists(CLASS_PATH):
            class_names = np.load(CLASS_PATH, allow_pickle=True)

        print("--> SUCCESS: Full trained model loaded successfully.")
        print("--> Classes:", class_names)

        return model, class_names

    except Exception:
        print("--> Error loading trained model:")
        traceback.print_exc()
        return None, class_names
def is_skin_image(img_bgr):
    try:
        # Reject tiny/corrupted images
        h, w = img_bgr.shape[:2]
        if h < 50 or w < 50:
            return False

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

        # More tolerant HSV range
        lower_hsv = np.array([0, 10, 0], dtype="uint8")
        upper_hsv = np.array([35, 255, 255], dtype="uint8")

        # More tolerant YCrCb range
        lower_ycrcb = np.array([0, 120, 70], dtype="uint8")
        upper_ycrcb = np.array([255, 180, 135], dtype="uint8")

        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        combined_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        skin_pixels = cv2.countNonZero(combined_mask)
        total_pixels = h * w

        # Reduced threshold from 20% → 5%
        skin_ratio = skin_pixels / total_pixels

        return skin_ratio >= 0.05

    except Exception:
        return False

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password, request.form.get('password')):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        if User.query.filter_by(username=username).first():
            flash('Username already exists.')
            return redirect(url_for('register'))
        new_user = User(username=username, password=generate_password_hash(request.form.get('password')))
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session: return redirect(url_for('login'))
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session: return redirect(url_for('login'))
    
    try:
        files = request.files.getlist('file')
        if not files or files[0].filename == '': 
            return redirect(url_for('dashboard'))
        
        active_model, active_classes = get_model()
        if active_model is None:
            return "<h3>Server Error: The AI model structure could not be parsed dynamically. Check server initialization logs.</h3>", 500
            
        batch_results = []
        dangerous_types = ['melanoma', 'basal cell carcinoma', 'actinic keratoses']

        for file in files:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None: continue
            
            _, buffer = cv2.imencode('.jpg', img_bgr)
            b64_str = base64.b64encode(buffer).decode('utf-8')
            
            if not is_skin_image(img_bgr):
                batch_results.append({
                    "filename": file.filename, "encoded_img": b64_str,
                    "status": "INVALID", "verdict": "INVALID: Not a skin image.",
                    "color_class": "bg-yellow-100 text-yellow-800 border-yellow-300"
                })
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (64, 64)).astype("float32") / 255.0
            tensor_input = np.expand_dims(img_resized, axis=0)
            
            preds = active_model.predict(tensor_input)[0]
            top_idx = np.argmax(preds)
            predicted_class_raw = str(active_classes[top_idx]).lower()
            confidence = round(float(preds[top_idx]) * 100, 2)
            
            # Filter 1: Catch uniform baseline/flat guesses
            if predicted_class_raw == "normal_skin":
                verdict = f"NO CANCER DETECTED (Confidence: {confidence}%)"
                color_class = "bg-green-100 text-green-800 border-green-300"

            elif predicted_class_raw in dangerous_types:
                verdict = f"CANCER RISK DETECTED (Type: \"{predicted_class_raw.upper()}\") (Confidence: {confidence}%)"
                color_class = "bg-red-100 text-red-800 border-red-300"

            else:
                verdict = f"BENIGN SKIN MARK (Type: \"{predicted_class_raw.upper()}\") (Confidence: {confidence}%)"
                color_class = "bg-green-100 text-green-800 border-green-300"
                
            batch_results.append({
                "filename": file.filename, "encoded_img": b64_str, "status": "VALID",
                "verdict": verdict, "color_class": color_class,
                "matrix": sorted([{"name": str(active_classes[i]), "prob": round(float(preds[i])*100, 1)} for i in range(len(active_classes))], key=lambda x: x['prob'], reverse=True)
            })

        tf.keras.backend.clear_session()
        return render_template('results.html', results=batch_results)
        
    except Exception as e:
        error_tracking = traceback.format_exc()
        return f"<h3>Inference Pipeline Crash Log:</h3><pre>{error_tracking}</pre>", 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))