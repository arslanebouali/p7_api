from Utils.predicter import PipelinePredictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
#import lime
import time
#from lime import lime_text
#import lime.lime_tabular
from flask import Flask, request, jsonify, render_template
from Utils.records import data
import pickle

full_model=('models/lgbm_full_model.sav')
model=('models/lgbm_model.sav')
explainer =('models/lime_explainer.pkl')
###function that return probability,  et lime explainer pour les deux modéles




app = Flask(__name__)

predictor = PipelinePredictor(model_path=model, explainer_path=explainer)


@app.route('/')
def home():
    return "Home"

Data= data()
@app.route('/predict/<id_client>', methods=['GET'])
def predict(id_client):

    client_full ,client_full_predict=Data.full_records(id_client)


    prediction,proba = predictor.predict(client_full_predict)
    explainer_model = predictor.explain(client_full)

    return prediction , proba , explainer_model



if __name__ == "__main__":
    predict(100009)
