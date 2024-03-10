import sys

sys.path.append("grocery-sales-forecasting")

from django.shortcuts import render

from tasks import make_inference, make_train


def home(request):
    return render(request, "index.html")


def get_predictions(store_number):
    task_result = make_inference.delay(store_number=store_number)

    return task_result.get()


def make_training():
    task_result = make_train.delay()

    return task_result.get()


def result(request):
    store_number = int(request.GET["store_number"])

    status = get_predictions(store_number=store_number)

    return render(request, "result.html", {"result": status})


def result_training(request):

    status = make_training()

    return render(request, "result_training.html", {"result": status})
