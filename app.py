import os
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Configure SQLite Database Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'database.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

os.makedirs(os.path.join(BASE_DIR, 'instance'), exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

model = None
class_names = None

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

with app.app_context():
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    if not inspector.has_table("user"):
        db.create_all()

# Dummy class to safely catch Keras 3 structural metadata under Keras 2
class PatchDTypePolicy:
    def __init__(self, *args, **kwargs): pass
    @classmethod
    def from_config(cls, config): return cls()

def get_model():
    global model, class_names
    if model is not None:
        return model, class_names

    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'skin_cancer_model.h5')
    CLASSES_PATH = os.path.join(BASE_DIR, 'models', 'class_names.npy')

    if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
        import tensorflow as tf
        from tensorflow.keras.models import load_model, Sequential
        from tensorflow.keras.layers import Input, Layer

        class SafeLayer(Layer):
            def __init__(self, *args, **kwargs):
                kwargs.pop('data_format', None)
                kwargs.pop('mode', None)
                kwargs.pop('seed', None)
                kwargs.pop('dtype', None) # Pop structural policy keys cleanly
                super().__init__(*args, **kwargs)
            def call(self, inputs):
                return inputs

        class_names = np.load(CLASSES_PATH, allow_pickle=True)
        
        try:
            custom_dict = {
                'InputLayer': lambda **kw: Input(shape=(64, 64, 3)),
                'RandomFlip': SafeLayer,
                'RandomRotation': SafeLayer,
                'RandomZoom': SafeLayer,
                'DTypePolicy': PatchDTypePolicy # Safely map out unexpected serialization policies
            }
            
            with tf.keras.utils.custom_object_scope(custom_dict):
                base_model = load_model(MODEL_PATH, compile=False)
            
            model = Sequential()
            model.add(Input(shape=(64, 64, 3)))
            for layer in base_model.layers:
                if layer.__class__.__name__ not in ['RandomFlip', 'RandomRotation', 'RandomZoom', 'InputLayer']:
                    model.add(layer)
            print("[Lazy Load] AI Model successfully built in memory.")
        except Exception as e:
            print(f"[Lazy Load Error] Failed to reconstruct: {e}")
            model = None
    else:
        class_names = ["Engine Error: Model files missing"]
        model = None
        
    return model, class_names

def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))  
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html', username=session['username'])
    return render_template('index.html', username=None)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the workspace.', 'warning')
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))
        
    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('dashboard'))
        
    file = request.files['file']
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('dashboard'))
        
    active_model, active_classes = get_model()
        
    if file and active_model is not None:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            processed_img = prepare_image(filepath)
            predictions = active_model.predict(processed_img)[0]
            
            if os.path.exists(filepath):
                os.remove(filepath)
                
            results = [
                {"class": str(active_classes[i]), "probability": round(float(predictions[i]) * 100, 2)}
                for i in range(len(active_classes))
            ]
            results = sorted(results, key=lambda x: x['probability'], reverse=True)
            
            import tensorflow as tf
            tf.keras.backend.clear_session() # Defensive RAM recovery step for Render instances
            
            return render_template('results.html', results=results, top_match=results[0], username=session['username'])
        except Exception as e:
            if os.path.exists(filepath): os.remove(filepath)
            flash(f"Inference execution failed: {str(e)}", "danger")
            return redirect(url_for('dashboard'))
        
    flash('Model processing engine initialization failed. Check your models folder logs.', 'danger')
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)