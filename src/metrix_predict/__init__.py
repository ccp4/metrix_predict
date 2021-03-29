import pkg_resources
import joblib


def get_data(filename):
    """Gets a data file from the package.

    Args:
        filename: The name of the data file as located in the package data
          directory.

    Returns:
        A readable file-like object for the data file.
    """

    path = "data/" + filename
    return pkg_resources.resource_stream(__name__, path)


# Load pre-determined classifier for this release
model = joblib.load(get_data("calibrated_classifier_20200408_1552.pkl"))


def predict(sample, model=model):
    """Predicts for sample data.

    Args:
        sample: A numpy array containing data processing metrics for datasets
          of interest. Each dataset consists of a separate row and the columns
          are in defined order: "lowreslimit", "anomalousslope", "anomalousCC",
          "diffI", "diffF", and "f".
        model: The model to use to generate predictions, set by default to a
          classifier distributed with this package.

    Returns:
        A tuple of arrays giving the prediction class (0 or 1), the probability
        of failure and the probability of success for each row of the input.
    """

    y_pred = model.predict(sample)
    y_pred_proba = model.predict_proba(sample)
    p_fail = y_pred_proba[:, 0]
    p_success = y_pred_proba[:, 1]
    return y_pred, p_fail, p_success


__all__ = ["get_data", "model", "predict"]
