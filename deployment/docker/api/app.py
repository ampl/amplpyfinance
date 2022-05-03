import os
import celery.states as states
from flask import Flask, Response, request, url_for, jsonify, make_response
from celery import Celery

dev_mode = True
app = Flask(__name__)


CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
CELERY_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://localhost:6379"
)

celery = Celery("tasks", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)


@app.route("/", methods=["GET", "POST"])
def home():
    response = make_response(request.url)
    response.mimetype = "text/plain"
    return response


@app.route("/solve", methods=["POST", "PUT"])
def solve() -> str:
    task = celery.send_task("tasks.solve", args=[request.json], kwargs={})
    return jsonify(
        {
            "url": request.url_root.rstrip("/")
            + url_for("check_task", task_id=task.id),
            "id": task.id,
        }
    )


@app.route("/check/<string:task_id>")
def check_task(task_id: str) -> str:
    res = celery.AsyncResult(task_id)
    if res.state == states.PENDING:
        return res.state
    else:
        try:
            return jsonify(res.result)
        except:
            return str(res.result)


@app.route("/health_check")
def health_check() -> Response:
    return jsonify("OK")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
