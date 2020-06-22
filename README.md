# Terrain Generator

This is a terrain generation library. It can generate infinitely sized random landscapes that use realistic simulation techniques such as erosion to make the landscapes more realistic. The library can be used for any 3D Python project such as a game with an open world and will eventually be available on [PyPI](https://pypi.org/). 

## Setup

This project uses [Poetry](https://python-poetry.org/) for dependency management. In order to set up your environment to use this package, execute the following commands:

```bash
poetry run pip install --upgrade pip
poetry install --no-dev
poetry shell
```

This should open a virtual environment which, upon running commands, will have access to the `terrain` Python package.