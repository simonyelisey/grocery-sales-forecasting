import os
import sys

from celery import Celery

sys.path.append("./grocery-sales-forecasting")
from infer import main as inference
from train import main as training

app = Celery("tasks_worker")

app.conf.broker_url = os.environ["REDIS_BROKER_URL"]
app.conf.result_backend = os.environ["REDIS_RESULT_BACKEND"]


@app.task(queue="train")
def make_train():
    training()
    inference()

    return "ok"


@app.task(queue="infer")
def make_inference():
    inference()

    return "ok"
