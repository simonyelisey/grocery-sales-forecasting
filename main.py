import datetime
import os
import sys

sys.path.append("./grocery-sales-forecasting")
from infer import main as inference
from train import main as training


def main():
    if datetime.datetime.now().day % 14 == 0 or "catboost.cbm" not in os.listdir(
        "./models"
    ):
        training()
    else:
        inference()


if __name__ == "__main__":
    main()
