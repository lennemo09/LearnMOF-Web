# LearnMOF-Web
Batch image inference and database interface for LearnMOF project - a joint colaboration for material discovery between UC Berkeley, Monash University, and VinUniversity.

## Run with Docker

- In root folder of the repo, run this command:

```shell
docker-compose up
```


## Prerequisites

- [Yarn](https://classic.yarnpkg.com/lang/en/docs/install/#mac-stable)
- Python 3.10
- [Poetry](https://python-poetry.org/)
- Docker
- [MongoDB](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/) 
- [pre-commit](https://pre-commit.com/)

## Setup

### Pre-commit

- Install pre-commit hooks

```shell
pre-commit install
```

### Frontend

- In the "frontend" folder of this repo, use this command to install all the dependencies 

```shell
yarn install
```

### Backend

- Go to `backend` folder

- Create virtual environment

```shell
python -m venv venv
```

- Activate virtual environment

```shell
. venv/bin/activate
```

- Install poetry: https://python-poetry.org/

- Install dependencies

```shell
poetry install
```

## Development

### Database

```shell
brew services start mongodb-community
```

### Backend

- Go to `backend` folder
- Activate virtual environment

```shell
source venv/bin/activate
```

- Run flask app

```shell
python run.py
```

- For more information, read README.md inside `backend` folder

### Frontend

- In `frontend` folder, run this command:

```shell
yarn start
```

