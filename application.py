from flask import Flask, request, render_template, jsonify
import numpy as np
import os
import cv2
from werkzeug.utils import secure_filename

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__, template_folder='src/templates')

app=application

UPLOAD_FOLDER='static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def extract_eye_region(filepath):
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        print("No face detected, using full image")
        return filepath  # fallback: send full image as-is

    # Take the largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_gray = gray[y:y+h, x:x+w]
    face_color = img[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    if len(eyes) == 0:
        print("No eyes detected, using top portion of face")
        eye_region = face_color[int(h * 0.15):int(h * 0.55), :]
    else:
        eyes = sorted(eyes, key=lambda e: e[0])
        ex, ey, ew, eh = eyes[0]
        eye_region = face_color[ey:ey+eh, ex:ex+ew]

    eye_path = filepath.replace('.jpg', '_eye.jpg')
    cv2.imwrite(eye_path, eye_region)
    print(f"Eye region saved to: {eye_path}")
    return eye_path

@app.route('/')
def index():
    return render_template('website.html')

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        if 'file' not in request.files:
            return jsonify({'error':'No file uploaded'}), 400
        file=request.files['file']

        if file.filename=='':
            return jsonify({'error':'No file selected'}), 400
        
        filename=secure_filename(file.filename)
        filepath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print(f"File saved to: {filepath}")

        eye_filepath = extract_eye_region(filepath)

        data=CustomData(image_path=eye_filepath)
        img_array=data.get_data_as_array()

        print(f"Images array shape: {img_array.shape}")

        predict_pipeline=PredictPipeline()
        pred_label, probabilities=predict_pipeline.predict(img_array)

        print(f"Prediction: {pred_label}")
        print(f"Probabilities: {probabilities}")

        classes=['Closed', 'no_yawm', 'Open', 'yawn']
        prob_dict={classes[i]: f"{probabilities[i]*100:.2f}%" for i in range(len(classes))}

        return jsonify({
            'prediction': pred_label,
            'probabilities': prob_dict
        })
    
    except Exception as e:
        print("ERROR:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__=="__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)