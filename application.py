from flask import Flask, request, render_template, jsonify
import numpy as np
import os
from werkzeug.utils import secure_filename

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__, template_folder='src/templates')

app=application

UPLOAD_FOLDER='static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

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

        data=CustomData(image_path=filepath)
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