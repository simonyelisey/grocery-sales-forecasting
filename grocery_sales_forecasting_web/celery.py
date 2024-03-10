import os

from celery import Celery

# Set the default Django settings module for the 'celery' program.
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "grocery_sales_forecasting_web.settings"
)

app = Celery("grocery_sales_forecasting_web")

app.conf.broker_url = os.environ["REDIS_BROKER_URL"]
app.conf.result_backend = os.environ["REDIS_RESULT_BACKEND"]

# Configure Celery using settings from Django settings.py.
app.config_from_object("django.conf:settings", namespace="CELERY")

# Load tasks from all registered Django app configs.
app.autodiscover_tasks()
