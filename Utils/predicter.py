import joblib
import dill

class PipelinePredictor():

    def __init__(self, model_path, explainer_path):
        self.model = joblib.load(model_path)
        self.explainer = joblib.load(explainer_path)

    def predict(self, x):
        # do pre-processing stuff

        pred = self.model.predict(x)
        proba = self.model.predict_proba(x)
        # do post-processing stuff
        return pred, proba


    def explain(self,record):
        explainer = self.explainer
        model = self.model
        exp = explainer.explain_instance(record, model.predict_proba)
        fig = exp


if __name__ == "__main__":
    full_model = ('models/lgbm_full_model.sav')
    model = ('models/lgbm_model.sav')
    explainer = ('models/lime_explainer.pkl')
    PipelinePredictor(model_path=model,explainer_path=explainer)
