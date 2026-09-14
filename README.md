# Lung Cancer Detection & Risk Assessment

An educational machine-learning web application with two screening workflows:

- CT image classification using a trained convolutional neural network (CNN)
- Questionnaire-based risk-level assessment using a trained neural-network model

The application is built with Flask and TensorFlow and includes its trained model files. It is a student research project, not a medical device and not a substitute for professional diagnosis.

## Features

- Upload a PNG or JPG CT image and receive the model's predicted class and confidence
- Complete a 23-input risk-factor form and receive a Low, Medium, or High model result
- Review educational lung-health information through the built-in assistant
- Use `/health` to verify that the application and both models loaded successfully
- Run with Flask locally or Gunicorn/Docker in production

## Technology

Python, Flask, TensorFlow/Keras, NumPy, pandas, scikit-learn, OpenCV, Pillow, Gunicorn and Docker.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`, then open `http://localhost:5000`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q tests/test_deployment_app.py
```

## Deployment

The repository contains a production Dockerfile and Render Blueprint. Set `SECRET_KEY` to a private random value.

## Important limitation

These predictions are educational model outputs, not clinically validated diagnoses. Anyone with medical concerns should consult a qualified healthcare professional.
