import datetime
import os
import sys

import hydra

sys.path.append("grocery-sales-forecasting")

from django.shortcuts import render
from infer import main as inference
from train import main as training


def home(request):
    return render(request, "index.html")


def get_predictions(store_number):
    if "catboost.cbm" not in os.listdir("./models"):
        training()
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        status = inference(store_number=store_number)
    else:
        status = inference(store_number=store_number)
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    return status


def make_training():
    training()
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    status = (
        f"{datetime.datetime.now()}, success train. Model is saved to /models folder."
    )

    return status


def result(request):
    store_number = int(request.GET["store_number"])

    status = get_predictions(store_number)

    return render(request, "result.html", {"result": status})


def result_training(request):

    status = make_training()

    return render(request, "result_training.html", {"result": status})
