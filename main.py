import datetime
import os
import sys

sys.path.append("./grocery-sales-forecasting")
import hydra

from tasks import make_inference as inference
from tasks import make_train as training


def main():
    if datetime.datetime.now().day % 14 == 0 or "catboost.cbm" not in os.listdir(
        "./models"
    ):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        training.delay()

    else:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        inference.delay()


if __name__ == "__main__":
    main()
