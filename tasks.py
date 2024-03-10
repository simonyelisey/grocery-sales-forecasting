import datetime
import os
import sys

import hydra

sys.path.append("./grocery-sales-forecasting")

from infer import main as inference
from train import main as training

from grocery_sales_forecasting_web import celery_app


@celery_app.task(queue="infer")
def make_inference(store_number):
    if "catboost.cbm" not in os.listdir("./models"):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        training()
        status = inference(store_number=store_number)
    else:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        status = inference(store_number=store_number)

    return status


@celery_app.task(queue="train")
def make_train():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    training()

    status = (
        f"{datetime.datetime.now()}, success train. Model is saved to /models folder."
    )

    return status
