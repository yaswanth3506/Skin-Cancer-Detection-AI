import os
import numpy as np
import cv2
import base64
import traceback
import sys
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "simple_secret_key"
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # Avoid large upload crashes

# Absolute paths for Linux compatibility
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
model = None
class_names = None

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class SafeLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(SafeLayer, self).__init__()
    def call(self, inputs):
        return inputs
    @classmethod
    def from_config(cls, config):
        return cls()

def get_model():
    global model, class_names
    if model is not None: 
        return model, class_names
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(current_dir, 'models', 'skin_cancer_model.h5')
    CLASSES_PATH = os.path.join(current_dir, 'models', 'class_names.npy')

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print(f"--> CRITICAL PATH ERROR: Cannot find files at:\n{MODEL_PATH}\nor\n{CLASSES_PATH}")
        return None, None

    from tensorflow.keras.models import load_model
    
    try:
        # NumPy 2.0 version mismatch patch
        if 'numpy._core' not in sys.modules:
            import numpy
            sys.modules['numpy._core'] = numpy

        class_names = np.load(CLASSES_PATH, allow_pickle=True)
        
        # Direct, safe configuration mapping dictionary
        custom_dict = {
            'RandomFlip': SafeLayer, 
            'RandomRotation': SafeLayer, 
            'RandomZoom': SafeLayer,
            'SafeLayer': SafeLayer
        }
        
        # Load the model directly using custom scopes to neutralize augmentation errors
        with tf.keras.utils.custom_object_scope(custom_dict):
            model = load_model(MODEL_PATH, compile=False)
            
        print("--> SUCCESS: AI Model and class names loaded completely!")
        return model, class_names
        
    except Exception as model_err:
        print("--> CORE AI LOADING ERROR DETAILS:")
        traceback.print_exc()
        return None, None

def is_skin_image(img_bgr):
    try:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        lower_hsv = np.array([0, 15, 0], dtype="uint8")
        upper_hsv = np.array([25, 170, 255], dtype="uint8")
        lower_ycrcb = np.array([0, 133, 77], dtype="uint8")
        upper_ycrcb = np.array([255, 173, 127], dtype="uint8")
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        combined_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        skin_pixels = cv2.countNonZero(combined_mask)
        total_pixels = img_bgr.shape[0] * img_bgr.shape[1]
        if total_pixels == 0 or (skin_pixels / total_pixels) < 0.20:
            return False
        return True
    except:
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
            return "<h3>Server Error: The AI model files could not be read or found on Render. Ensure models/ is pushed.</h3>", 500
            
        batch_results = []

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
            
            if 'normal' in predicted_class_raw or 'healthy' in predicted_class_raw:
                verdict = f"NO CANCER DETECTED (Confidence: {confidence}%)"
                color_class = "bg-green-100 text-green-800 border-green-300"
            else:
                verdict = f"CANCER DETECTED (Type: \"{predicted_class_raw.upper()}\") (Confidence: {confidence}%)"
                color_class = "bg-red-100 text-red-800 border-red-300"
                
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