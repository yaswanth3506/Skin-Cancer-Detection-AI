import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf

app = Flask(__name__)
app.secret_key = "supersecretkey"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
model = None
class_names = None

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class SafeLayer(object):
    def __init__(self, *args, **kwargs): pass
    @classmethod
    def from_config(cls, config): return cls()
    def __call__(self, inputs): return inputs

def get_model():
    global model, class_names
    if model is not None: return model, class_names
    
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'skin_cancer_model.h5')
    CLASSES_PATH = os.path.join(BASE_DIR, 'models', 'class_names.npy')

    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        from tensorflow.keras.models import load_model, Sequential
        from tensorflow.keras.layers import Input
        
        class_names = np.load(CLASSES_PATH, allow_pickle=True)
        custom_dict = {
            'InputLayer': lambda **kw: Input(shape=(64, 64, 3)),
            'RandomFlip': SafeLayer, 'RandomRotation': SafeLayer,
            'RandomZoom': SafeLayer, 'DTypePolicy': SafeLayer
        }
        with tf.keras.utils.custom_object_scope(custom_dict):
            base_model = load_model(MODEL_PATH, compile=False)
        
        model = Sequential([Input(shape=(64, 64, 3))])
        for layer in base_model.layers:
            if layer.__class__.__name__ not in ['RandomFlip', 'RandomRotation', 'RandomZoom', 'InputLayer', 'SafeLayer']:
                model.add(layer)
    return model, class_names

def prepare_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64)).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        new_user = User(username=username, password=generate_password_hash(password))
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
    files = request.files.getlist('file')
    if not files or files[0].filename == '': return redirect(url_for('dashboard'))
    
    active_model, active_classes = get_model()
    all_predictions = []

    for file in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        preds = active_model.predict(prepare_image(filepath))[0]
        os.remove(filepath)
        
        res = sorted([{"class": str(active_classes[i]), "probability": round(float(preds[i])*100, 2)} 
                      for i in range(len(active_classes))], key=lambda x: x['probability'], reverse=True)
        all_predictions.append({"filename": file.filename, "top_match": res[0], "full_matrix": res})

    tf.keras.backend.clear_session()
    return render_template('results.html', batch_results=all_predictions, username=session.get('username'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, 'instance'), exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)