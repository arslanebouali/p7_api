from Utils.predicter import PipelinePredictor
from flask import Flask, request, jsonify, render_template
from Utils.records import data
from Utils.Tools import *
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




    dict={"proba": proba.tolist(),
          "prediction " : prediction.tolist()}

    resp = jsonify(dict)
    resp.status_code = 200
    print(resp)
    return resp

from flask import send_file

@app.route('/explain/<id_client>')
def get_image(id_client):
    client_full, client_full_predict = Data.full_records(id_client)

    exp=predictor.explanation(client_full)
    predictor.explain(exp)
    filename= 'data/explanation.txt'
    return send_file(filename, mimetype='html')

@app.route('/df_maps/<id_client>', methods=['GET'])
def get_maps(id_client):
    client_full, client_full_predict = Data.full_records(id_client)
    exp = predictor.explanation(client_full)
    df_map_filtered_plus, df_map_filtered_neg = model_local_interpretation(exp)
    df = Data.Target

    df_map_filtered_plus = fonction_comparaison(df_map_filtered_plus, df, id_client, df)
    df_map_filtered_neg = fonction_comparaison(df_map_filtered_neg, df, id_client, df)

    dict = {"filtered_plus": df_map_filtered_plus.to_json(),
            "filtered_neg": df_map_filtered_neg.to_json()}

    resp = jsonify(dict)

    print(resp)
    return resp

if __name__ == "__main__":
    app.run(debug=True)
