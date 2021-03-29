import pytest
import pandas as pd
import metrix_predict


def test_metrix_predict():

    # Load sample data and extract array of interest
    data = pd.read_csv(metrix_predict.get_data("sample.csv"))
    data_initial = data[
        ["lowreslimit", "anomalousslope", "anomalousCC", "diffI", "diffF", "f"]
    ]
    data_initial = data_initial.fillna(0)
    data_initial = data_initial.fillna(0)
    unknown = data_initial.to_numpy()

    # Predict classes and probabilities
    classes, p_fail, p_success = metrix_predict.predict(unknown)

    # Check results are as expected for the sample data
    assert classes.tolist() == [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        1,
        1,
    ]
    assert p_fail == pytest.approx(
        [
            0.02155452,
            0.07697223,
            0.22951946,
            0.14439545,
            0.03376021,
            0.01346777,
            0.47083102,
            0.575352,
            0.61118045,
            0.56728882,
            0.44013163,
            0.76756068,
            0.55942766,
            0.77738948,
            0.88082336,
            0.43012523,
            0.77182828,
            0.93833842,
            0.64919585,
            0.21999304,
            0.49948678,
            0.80728087,
            0.48463023,
            0.43844386,
        ]
    )
    assert p_success == pytest.approx(
        [
            0.97844548,
            0.92302777,
            0.77048054,
            0.85560455,
            0.96623979,
            0.98653223,
            0.52916898,
            0.424648,
            0.38881955,
            0.43271118,
            0.55986837,
            0.23243932,
            0.44057234,
            0.22261052,
            0.11917664,
            0.56987477,
            0.22817172,
            0.06166158,
            0.35080415,
            0.78000696,
            0.50051322,
            0.19271913,
            0.51536977,
            0.56155614,
        ]
    )
