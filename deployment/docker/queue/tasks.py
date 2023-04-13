import os
from utils import run
from celery import Celery, signals
from efficient_frontier import solve_ef

CELERY_BROKER_URL = (os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379"),)
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379"
)

celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


@celery.task(name="tasks.solve")
def solve(data) -> dict:
    return solve_ef(data)


@signals.worker_process_init.connect
def on_worker_process_init(**kwargs):
    from amplpy import modules

    # Activate the license when the worker starts
    uuid = os.environ.get("AMPLKEY_UUID", None)
    if uuid is not None and uuid != "":
        modules.activate(uuid)
    modules.run(["ampl", "-vvq"], verbose=True)
