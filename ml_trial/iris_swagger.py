import pickle
from flask import Flask, request
from flasgger import Swagger
import numpy as np
import pandas as pd

with open('model_svc.pkl', 'rb') as model_svc_pkl:
    model_svc = pickle.load(model_svc_pkl)

with open('model_rfc.pkl', 'rb') as model_rfc_pkl:
    model_rfc = pickle.load(model_rfc_pkl)


ml_api = Flask(__name__)
swagger = Swagger(ml_api)

@ml_api.route('/predict_svc')
def predict_svc():
    """Endpoint to predict the species of Iris [0: Setosa', 1: 'Versicolor', 3: 'Virginica] using Support Vector Machine
       A 'get' implementation using Support Vector Machine 
    ---
    parameters:
      - name: sepal_length
        in: query
        type: number
        required: true
      - name: sepal_width
        in: query
        type: number
        required: true
      - name: petal_length
        in: query
        type: number
        required: true
      - name: petal_width
        in: query
        type: number
        required: true
    """
    sepal_length = request.args.get("sepal_length")
    sepal_width = request.args.get("sepal_width")
    petal_length = request.args.get("petal_length")
    petal_width = request.args.get("petal_width")

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model_svc.predict(input_data)
    return str(prediction)


@ml_api.route('/predict_rfc')
def predict_rfc():
    """Endpoint to predict the species of Iris flower [0: Setosa', 1: 'Versicolor', 3: 'Virginica] using Random Forest Classifier
       A 'get' implementation using Randon Forest Classifier
    ---
    parameters:
      - name: sepal_length
        in: query
        type: number
        required: true
      - name: sepal_width
        in: query
        type: number
        required: true
      - name: petal_length
        in: query
        type: number
        required: true
      - name: petal_width
        in: query
        type: number
        required: true
    """
    sepal_length = request.args.get("sepal_length")
    sepal_width = request.args.get("sepal_width")
    petal_length = request.args.get("petal_length")
    petal_width = request.args.get("petal_width")

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model_rfc.predict(input_data)
    return str(prediction)


@ml_api.route('/predict_svc_file', methods=["POST"])
def predict_svc_file():
    """Endpoint to predict the species of Iris flower using Support Vector Machine using file input
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    input_data = pd.read_csv(request.files.get("input_file"))
    prediction = model_svc.predict(input_data)
    return str(list(prediction))


@ml_api.route('/predict_rfc_file', methods=["POST"])
def predict_rfc_file():
    """Endpoint to predict the species of Iris flower using Random Forest Classifier using file input
    ---
    parameters:
      - name: input_file
        in: formData
        type: file
        required: true
    """
    input_data = pd.read_csv(request.files.get("input_file"))
    prediction = model_rfc.predict(input_data)
    return str(list(prediction))


if __name__ == '__main__':
    ml_api.run(host='0.0.0.0', port=8888)
