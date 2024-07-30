# mle-learning-python


## Setup

- Using PIP since torchtyping must be installed with pip

```bash
conda create -n pytorch-stable-pip python
```

- Activate enviroment

```bash
conda activate pytorch-stable-pip
```

- Install the rest using pip

```bash
pip install -r requirements.txt
```

## Running tests

- Run all

```bash
pytest -v -s --disable-warnings
```
