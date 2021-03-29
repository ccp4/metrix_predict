import metrix_predict
import argparse
import pandas as pd
import sys


def probability_type(arg):
    """Type function for argparse.

    Args:
        arg: An object convertible to float in the range [0,1].

    Returns:
        The converted float in the range [0,1].

    Raises:
        argparse.ArgumentTypeError: An error occurred converting the argument
          to float in the range [0,1].
    """

    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number in [0,1]")
    if f < 0.0 or f > 1.0:
        raise argparse.ArgumentTypeError("Argument must be between 0.0 and 1.0")
    return f


def main():
    """Entry point function for the metrix_predict program."""

    parser = argparse.ArgumentParser(
        description="Predict experimental phasing success."
    )
    parser.add_argument(
        "csv_file", help="Path to a .csv formatted file containing the required metrics"
    )
    parser.add_argument(
        "--cutoff",
        type=probability_type,
        default=0.80,
        help="Probability cutoff for determining the adjusted class",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        nargs="?",
        type=argparse.FileType("w"),
        help="output CSV format file",
    )

    model = metrix_predict.model

    args = parser.parse_args()
    try:
        data = pd.read_csv(args.csv_file)
    except Exception:
        sys.exit(f"Unable to read CSV data from {args.csv_file}")

    try:
        data_initial = data[
            ["lowreslimit", "anomalousslope", "anomalousCC", "diffI", "diffF", "f"]
        ]
    except KeyError as e:
        sys.exit(f"Required data not found: {e}")

    data_initial = data_initial.fillna(0)
    unknown = data_initial.to_numpy()

    data["Class"], data["P(fail)"], data["P(success)"] = metrix_predict.predict(unknown)
    data["Adj. class"] = (data["P(success)"] >= args.cutoff).astype(int)

    if args.outfile:
        print(f"Writing to {args.outfile.name}")
        data.to_csv(args.outfile, index=False, float_format="%g")
    else:
        print(data)
        print(f"\nAdj. class is determined by the cutoff p(success) >= {args.cutoff}")
