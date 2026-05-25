from flask import Flask, render_template, request, redirect, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import os
import uuid
from PIL import Image
from utils import predict_image, is_skin_image

app = Flask(__name__)

# Core Settings
app.config['SECRET_KEY'] = 'skin_cancer_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    if 'user' in session:
        return render_template('home.html', username=session['user'])
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return "Username already exists"
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user'] = username
            return redirect('/')
        else:
            return "Invalid username or password"
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect('/login')

    prediction_results = []

    if request.method == 'POST':
        files = request.files.getlist('images')

        if not files or files[0].filename == '':
            return "No files selected. Please upload a valid PNG or JPEG image."

        for file in files:
            if file.filename != '':
                if not allowed_file(file.filename):
                    prediction_results.append({
                        'filename': file.filename,
                        'unique_filename': '',
                        'prediction': "Invalid Image (Unsupported format)",
                        'probability': 0.0,
                        'next_prediction': 'N/A',
                        'next_probability': 0.0
                    })
                    continue

                unique_filename = str(uuid.uuid4()) + "_" + file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)

                try:
                    with Image.open(filepath) as img:
                        img.verify()
                except Exception:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    prediction_results.append({
                        'filename': file.filename,
                        'unique_filename': '',
                        'prediction': "Invalid Image (Corrupted file)",
                        'probability': 0.0,
                        'next_prediction': 'N/A',
                        'next_probability': 0.0
                    })
                    continue

                if not is_skin_image(filepath):
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    prediction_results.append({
                        'filename': file.filename,
                        'unique_filename': '',
                        'prediction': "Invalid Image",
                        'probability': 0.0,
                        'next_prediction': 'N/A',
                        'next_probability': 0.0
                    })
                    continue

                # Run inference tracking both highest options
                prediction, probability, next_pred, next_prob = predict_image(filepath)

                if prediction == "normal_skin":
                    display_pred = "No Cancer"
                else:
                    display_pred = f"CANCER (type={prediction})"

                if next_pred == "normal_skin":
                    display_next = "No Cancer"
                else:
                    display_next = f"CANCER (type={next_pred})"

                prediction_results.append({
                    'filename': file.filename,
                    'unique_filename': unique_filename,
                    'prediction': display_pred,
                    'probability': round(probability, 2),
                    'next_prediction': display_next,
                    'next_probability': round(next_prob, 2)
                })

    return render_template('upload.html', prediction_results=prediction_results, username=session['user'])

@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    with app.app_context():
        db.create_all()
    app.run(debug=True)