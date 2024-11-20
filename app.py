from flask import Flask, render_template, request, redirect, url_for, session, flash
import cv2
import numpy as np
from skimage import restoration
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
import os
from werkzeug.utils import secure_filename

# Flask setup
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Upload folder setup
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Database and model setup
users_db = {}  # Simulated database
cnn_model = load_model('path_to_cnn_model.h5')  # Load the pre-trained CNN model

# Helper functions
def enhance_image(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    enhanced_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return enhanced_img

def restore_image(image):
    restored_img = restoration.denoise_nl_means(image, multichannel=True)
    return (restored_img * 255).astype(np.uint8)

def segment_image(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, segmented_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
    return segmented_img

def morphology_image(image):
    kernel = np.ones((5, 5), np.uint8)
    morph_img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return morph_img

def extract_features_with_cnn(image):
    resized_image = cv2.resize(image, (160, 160))  # Resize to CNN input size
    image = np.expand_dims(resized_image, axis=0) / 255.0  # Normalize
    embeddings = cnn_model.predict(image)
    return embeddings[0]

def compare_features(stored_features, input_features):
    similarity = cosine_similarity([stored_features], [input_features])
    return similarity[0][0] > 0.9  # Matching threshold

# Routes
@app.route('/')
def home():
    return render_template_string(index_html)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        account_number = request.form['account_number']
        pin = request.form['pin']
        file = request.files['photo']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process image and extract features
            image = cv2.imread(filepath)
            image = enhance_image(image)
            image = restore_image(image)
            image = segment_image(image)
            image = morphology_image(image)
            features = extract_features_with_cnn(image)
            
            # Save user data
            users_db[username] = {
                "account_number": account_number,
                "pin": pin,
                "features": features
            }
            os.remove(filepath)  # Clean up uploaded image
            flash("Signup successful! You can now log in.", "success")
            return redirect(url_for('login'))
    return render_template_string(signup_html)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        pin = request.form['pin']
        file = request.files['photo']
        if username in users_db and users_db[username]['pin'] == pin:
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Process image and extract features
                image = cv2.imread(filepath)
                image = enhance_image(image)
                image = restore_image(image)
                image = segment_image(image)
                image = morphology_image(image)
                features = extract_features_with_cnn(image)
                
                # Compare features
                stored_features = users_db[username]['features']
                if compare_features(stored_features, features):
                    session['username'] = username
                    os.remove(filepath)  # Clean up uploaded image
                    return redirect(url_for('dashboard'))
                else:
                    flash("Face verification failed.", "danger")
        flash("Invalid credentials or face verification failed.", "danger")
    return render_template_string(login_html)

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        username = session['username']
        return f"Welcome to your dashboard, {username}!"
    return redirect(url_for('login'))

# Inline HTML templates
index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATM Face Recognition</title>
</head>
<body>
    <h1>Welcome to ATM Face Recognition</h1>
    <p>Secure your transactions with facial recognition technology.</p>
    <a href="{{ url_for('signup') }}">Signup</a>
    <a href="{{ url_for('login') }}">Login</a>
</body>
</html>
'''

signup_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signup</title>
</head>
<body>
    <h1>Signup</h1>
    <form method="post" enctype="multipart/form-data">
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" required><br>
        <label for="account_number">Account Number:</label>
        <input type="text" id="account_number" name="account_number" required><br>
        <label for="pin">PIN:</label>
        <input type="password" id="pin" name="pin" required><br>
        <label for="photo">Upload Photo:</label>
        <input type="file" id="photo" name="photo" accept="image/*" required><br>
        <button type="submit">Signup</button>
    </form>
    <a href="{{ url_for('home') }}">Back to Home</a>
</body>
</html>
'''

login_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login</title>
</head>
<body>
    <h1>Login</h1>
    <form method="post" enctype="multipart/form-data">
        <label for="username">Username:</label>
        <input type="text" id="username" name="username" required><br>
        <label for="pin">PIN:</label>
        <input type="password" id="pin" name="pin" required><br>
        <label for="photo">Upload Photo:</label>
        <input type="file" id="photo" name="photo" accept="image/*" required><br>
        <button type="submit">Login</button>
    </form>
    <a href="{{ url_for('home') }}">Back to Home</a>
</body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True)
