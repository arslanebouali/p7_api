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

    def explanation(self,record):
        explainer = self.explainer
        model = self.model
        exp = explainer.explain_instance(record, model.predict_proba)
        return exp
        
    def explain(self,exp):
        exp.save_to_file('data/explanation.txt', labels=None, predict_proba=False, show_predicted_value=False)
        print("explanation is ready ")

if __name__ == "__main__":
    full_model = ('models/lgbm_full_model.sav')
    model = ('models/lgbm_model.sav')
    explainer = ('models/lime_explainer.pkl')


