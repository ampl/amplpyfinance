```{eval-rst}
.. _deployment:
```
# Deployment

In order to run applications independently of the computing environment, containerization
is the most common solution. There are many ways to deploy containerized applications.
Here we provide examples and tips to help you getting started.

## Docker Compose

[Docker Compose](https://docs.docker.com/compose/) is a very popular tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application’s services.

Using Compose is basically a three-step process:
- Define your app's environment with a `Dockerfile` so it can be reproduced anywhere.
- Define the services that make up your app in `docker-compose.yml` so they can be run together in an isolated environment.
- Run `docker compose up` and the Docker compose command starts and runs your entire app.

### Deployment with Docker Compose

A basic [Docker Compose](https://docs.docker.com/compose/) template for orchestrating a [Flask](http://flask.pocoo.org/) application & a [Celery](http://www.celeryproject.org/) queue with [Redis](https://redis.io/) can be found at [amplpyfinance/deployment/docker](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker).

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
$ cd amplpyfinance/deployment/docker
$ export COMPOSE_PROJECT_NAME=amplpyfinance
$ docker compose up -d --build
```

### Enable hot code reload

In addition to [docker-compose.yml](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/docker-compose.yml), we provide [dev.yml](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/docker-compose.yml) to override some specifications in `docker-compose.yml` to address development needs such as hot code reload.

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

To change the endpoints, update the code in [api/app.py](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/api/app.py)

Task changes should happen in [queue/tasks.py](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/queue/tasks.py)

## Submit jobs to the containerized application

After running `docker compose up` check the endpoints with `docker compose ps` to make sure they are running and to see the ports being used.

You can then use [client.py](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/client.py) with `python client.py 127.0.0.1:80` to submit jobs. When running locally the IP address is `127.0.0.1` and the port should be `80` or `8080` (if `dev.yml` is being used).

You can check the status of the jobs by accessing the [Flower](https://github.com/mher/flower) server that uses port `5555`. You can check in your browser `http://127.0.0.1:5555/` when running locally.

## Deploying on AWS using Docker Compose

[Deploying Docker containers on ECS](https://docs.docker.com/cloud/ecs-integration/) can be done with docker compose as follows:

- First you need to build and publish your docker images:
  - Build the images with `docker compose build`. Note: you will need to rename the images in [docker-compose.yml](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/docker-compose.yml) in order to upload to your own account.
  - Upload the docker images `docker compose push`.
- You can then deploy the application as follows:
  - Create a docker context for ECS with `docker context create ecs myecscontext`.
  - Select the context to use with `docker context use myecscontext`.
  - Deploy with `docker compose up -d`.
  - Check the endpoints with `docker compose ps` in order to know the the IP addresses and ports being used.
- Stop the deployed application `docker compose down` (if you forget this step the application will keep running and you will keep being charged for the resources being used). 

You can get to the default context (i.e., local deployment) with `docker context use default`.

You can read more about this type of deployment at [Docker Compose: From Local to Amazon ECS](https://www.docker.com/blog/docker-compose-from-local-to-amazon-ecs/).

## Deploying on Azure using Docker Compose

[Deploying Docker containers on Azure](https://docs.docker.com/cloud/aci-integration/) can be done with docker compose as follows:

- First you need to build and publish your docker images:
  - Build the images with `docker compose build`. Note: you will need to rename the images in [docker-compose.yml](https://github.com/ampl/amplpyfinance/tree/master/deployment/docker/docker-compose.yml) in order to upload to your own account.
  - Upload the docker images `docker compose push`.
- You can then deploy the application as follows:
  - Create a docker context for ECS with `docker context create aci myacicontext`.
  - Select the context to use with `docker context use myacicontext`.
  - Deploy with `docker compose up -d`.
  - Check the endpoints with `docker compose ps` in order to know the the IP addresses and ports being used.
- Stop the deployed application `docker compose down` (if you forget this step the application will keep running and you will keep being charged for the resources being used). 

You can get to the default context (i.e., local deployment) with `docker context use default`.
