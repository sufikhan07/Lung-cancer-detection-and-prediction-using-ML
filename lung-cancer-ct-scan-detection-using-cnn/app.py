from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

app = Flask(__name__)
# Use an absolute uploads directory inside the project so we can serve files from it reliably
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model("lung_cancer_cnn_model.keras")
IMG_SIZE = (224, 224)
# default label order (index 0, index 1) - can be overridden by a saved mapping
class_labels = ['Cancerous', 'Non-Cancerous']

# Try to load class index mapping produced at training time (optional)
# Expected format (example): {"Cancerous": 0, "Non-Cancerous": 1}
index_to_class = {}
try:
    mapping_path = os.path.join(app.root_path, 'class_indices.json')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            cls_map = json.load(f)
        # invert mapping to get index -> class name
        for name, idx in cls_map.items():
            index_to_class[int(idx)] = name
except Exception:
    index_to_class = {}

# fallback to defaults if mapping not found
if not index_to_class:
    index_to_class = {0: class_labels[0], 1: class_labels[1]}

def predict(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    if prediction.shape[-1] == 1:
        # For a single-sigmoid output Keras/flow_from_directory convention:
        # the sigmoid output is the probability for class index 1.
        prob = float(prediction[0][0])
        if prob >= 0.5:
            class_idx = 1
            confidence = prob
        else:
            class_idx = 0
            confidence = 1.0 - prob
        label = index_to_class.get(class_idx, class_labels[class_idx])
        return label, confidence
    elif prediction.shape[-1] == 2:
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx])
        return class_labels[class_idx], confidence
    return "Unknown", 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        contact = request.form.get('contact')
        img_file = request.files.get('ct_image')

        if not name or not age or not contact or not img_file:
            return render_template('form.html', error="Please fill all fields and upload image.")

        img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
        img_file.save(img_path)

        result, conf = predict(img_path)
        return render_template(
            'result.html',
            name=name,
            age=age,
            contact=contact,
            result=result,
            confidence=f"{conf * 100:.2f}%",
            image_url=url_for('uploaded_file', filename=img_file.filename)
        )
    return render_template('form.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Serve files from the configured uploads directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)