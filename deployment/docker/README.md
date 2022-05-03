# Quick deployment with Docker, Flask, Celery, and Redis

A basic [Docker Compose](https://docs.docker.com/compose/) template for orchestrating a [Flask](http://flask.pocoo.org/) application & a [Celery](http://www.celeryproject.org/) queue with [Redis](https://redis.io/).

Jobs submitted to the flask application are sent to the queue and then processed by one of the optimization workers (docker containers with AMPL and solvers installed) where [amplpyfinance](https://github.com/ampl/amplpyfinance) is used to solve optimization problems.

### Project structure

- [docker-compose.yml](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/docker-compose.yml): definition of the services involved in this Docker application.
- [api/](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/api/)
  - [Dockerfile](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/api/Dockerfile): Dockerfile for the Flask application.
  - [app.py](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/api/app.py): Flask application.
  - [requirements.txt](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/api/requirements.txt): Python requirements for the Flask application.
- [queue/](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/queue/)
  - [Dockerfile](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/queue/Dockerfile): Dockerfile for Celery and its workers.
    - This Dockerfile includes the steps to install AMPL and solvers since this needs to be available in the workers.
  - [tasks.py](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/queue/tasks.py): definition of Celery tasks and startup handler (used to activate the AMPL license when a new worker starts).
  - [requirements.txt](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/queue/requirements.txt): Python requirements for the Celery queue and workers.
- [client.py](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/client.py): Python client to submit requests to the application.

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
