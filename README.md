# fed_ai

## Installation and Setup

The project leverages [uv](https://docs.astral.sh/uv/) for dependency management and package building adn [ruff](https://docs.astral.sh/ruff/) for linting and code formatting. 

### 1. Install uv

Follow the instructions [here](https://docs.astral.sh/uv/getting-started/installation/)

Add the following line to your `.bashrc`:

```sh
export UV_NATIVE_TLS=TRUE
```

### 2. Clone the repository

### 3. Create virtual environment and install dependencies

```sh
uv sync
```

For following commands make sure, that the virtual environment is activated. Alternatively prepend commands with `uv run`

### 4. Make sure Docker Daemon is running

### 5. Make sure Docker Compose V2 is installed

## Run Project

```sh
export FLWR_VERSION="1.19.0" # update with your version
docker-compose up --build -d
flwr run . local-deployment --stream
```

To read the logs, use 
```docker-compose logs -f```

At the end, to clean up:
```docker-compose down```
