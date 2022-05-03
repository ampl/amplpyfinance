# Quick deployment with Docker, Flask, Celery, and Redis

A basic [Docker Compose](https://docs.docker.com/compose/) template for orchestrating a [Flask](http://flask.pocoo.org/) application & a [Celery](http://www.celeryproject.org/) queue with [Redis](https://redis.io/).

Jobs submitted to the flask application are sent to the queue and then processed by one of the optimization workers (docker containers with AMPL and solvers installed) where [amplpyfinance](https://github.com/ampl/amplpyfinance) is used to solve optimization problems.

### Installation

```bash
git clone https://github.com/ampl/amplpyfinance
```

### AMPL and solver licenses

You can use the demo version that is subject to size limits or any cloud license (including AMPL CE licenses). You can specify the license by setting the environment variable `AMPLKEY_UUID`.

```
export AMPLKEY_UUID=xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx
```

You can also create a .env file in this directory with environment variables such as `AMPLKEY_UUID`.

```
AMPLKEY_UUID=xxxxxxxx-xxxx-Mxxx-Nxxx-xxxxxxxxxxxx
COMPOSE_PROJECT_NAME=amplpyfinance
...
```

### Build & Launch

```bash
$ cd amplpyfinance/deployment
$ export COMPOSE_PROJECT_NAME=amplpyfinance
$ docker compose up -d --build
```

### Enable hot code reload

```
$ export COMPOSE_PROJECT_NAME=amplpyfinance
$ docker compose -f docker-compose.yml -f dev.yml up --build
```

This will expose the Flask application's endpoints on port `8080` as well as a [Flower](https://github.com/mher/flower) server for monitoring workers on port `5555`

To add more workers:
```bash
$ export COMPOSE_PROJECT_NAME=amplpyfinance
$ docker compose up -d --scale worker=5 --no-recreate
```

To shut down:

```bash
$ docker compose down
```

To change the endpoints, update the code in [api/app.py](api/app.py)

Task changes should happen in [queue/tasks.py](queue/tasks.py) 

---

Adapted from [https://github.com/mattkohl/docker-flask-celery-redis](https://github.com/mattkohl/docker-flask-celery-redis) and [https://github.com/itsrifat/flask-celery-docker-scale](https://github.com/itsrifat/flask-celery-docker-scale)
