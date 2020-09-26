import pickle
from flask import Flask, request
import numpy as np
import pandas as pd

with open('model_svc.pkl', 'rb') as model_svc_pkl:
    model_svc = pickle.load(model_svc_pkl)

with open('model_rfc.pkl', 'rb') as model_rfc_pkl:
    model_rfc = pickle.load(model_rfc_pkl)


ml_api = Flask(__name__)

@ml_api.route('/predict_svc', methods=["GET"])
def predict_svc():
    sepal_length = request.args.get("sepal_length")
    sepal_width = request.args.get("sepal_width")
    petal_length = request.args.get("petal_length")
    petal_width = request.args.get("petal_width")

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model_svc.predict(input_data)
    return str(prediction)


@ml_api.route('/predict_rfc', methods=["GET"])
def predict_rfc():
    sepal_length = request.args.get("sepal_length")
    sepal_width = request.args.get("sepal_width")
    petal_length = request.args.get("petal_length")
    petal_width = request.args.get("petal_width")

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model_rfc.predict(input_data)
    return str(prediction)


@ml_api.route('/predict_svc_file', methods=["POST"])
def predict_svc_file():
    input_data = pd.read_csv(request.files.get("input_file"))
    prediction = model_svc.predict(input_data)
    return str(list(prediction))


@ml_api.route('/predict_rfc_file', methods=["POST"])
def predict_rfc_file():
    input_data = pd.read_csv(request.files.get("input_file"))
    prediction = model_rfc.predict(input_data)
    return str(list(prediction))


if __name__ == '__main__':
    ml_api.run(host='0.0.0.0', port=8888)
