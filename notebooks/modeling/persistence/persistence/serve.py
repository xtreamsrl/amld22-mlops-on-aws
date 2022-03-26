#!/usr/bin/env python

import os
from io import StringIO

import flask
import joblib
import pandas as pd
from flask import Flask, Response
from pandas import DatetimeIndex
from sktime.forecasting.base import ForecastingHorizon

model_dir = '/opt/ml/model'
model = joblib.load(os.path.join(model_dir, "model.joblib"))

app = Flask(__name__)

import warnings

warnings.filterwarnings("ignore")


@app.route("/ping", methods=["GET"])
def ping():
    return Response(response="\n", status=200)


@app.route("/invocations", methods=["POST"])
def predict():
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO(data)
        data = pd.read_csv(s, header=None)
        data = list(data.loc[0])

        fh = ForecastingHorizon(DatetimeIndex(data, freq='D'), is_relative=False)
        response = model.predict(fh)
        response = response.to_json()
    else:
        return flask.Response(response='CSV data only', status=415, mimetype='text/plain')

    return Response(response=response, status=200)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)