import pkg_resources
import joblib


def get_data(filename):
    path = "data/" + filename
    return pkg_resources.resource_stream(__name__, path)


model = joblib.load(get_data("calibrated_classifier_20200408_1552.pkl"))


def predict(sample, model=model):

    y_pred = model.predict(sample)
    y_pred_proba = model.predict_proba(sample)
    p_fail = y_pred_proba[:, 0]
    p_success = y_pred_proba[:, 1]
    return y_pred, p_fail, p_success


__all__ = ["get_data", "model", "predict"]
