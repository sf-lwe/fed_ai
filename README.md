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
cd fed_ai_setup
export PROJECT_DIR=quickstart-compose
docker compose up --build -d
flwr run quickstart-compose local-deployment --stream
```

At the end, to clean up:
```docker compose down -v```
