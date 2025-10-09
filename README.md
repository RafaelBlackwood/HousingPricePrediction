# Housing Price Prediction Mini

A small end to end project that explores regression models for predicting housing prices. It includes data preparation, modeling, evaluation, and simple ways to run the code locally. The repository also contains a short written report and presentation slides with the main findings. ([GitHub][1])

---

## Table of contents

* [Project goals](#project-goals)
* [Repository structure](#repository-structure)
* [Getting started](#getting-started)
* [Usage](#usage)
* [Configuration](#configuration)
* [Experiments and results](#experiments-and-results)
* [Troubleshooting](#troubleshooting)
* [Roadmap](#roadmap)
* [License](#license)
* [Acknowledgments](#acknowledgments)

---

## Project goals

* Load a tabular housing dataset and clean it
* Engineer features and split data into train and test
* Train baseline and improved regression models
* Evaluate with common metrics like MAE and RMSE
* Save artifacts so results are reproducible

---

## Repository structure

```
.
├── pythonProject/              # Source code for data prep and modeling
├── report.docx                 # Short report with method and results
├── Project.docx                # Project write up notes
├── HousePricingPrediction-Rafael A..pptx   # Presentation slides
└── .idea/                      # IDE project settings
```

Documents listed above are already in the repo. You can read the DOCX report and the PPTX slides for a quick overview of the approach and results. ([GitHub][1])

---

## Getting started

### Prerequisites

* Python 3.9 or newer
* pip
* A terminal with git installed

### Set up a virtual environment

```bash
git clone https://github.com/RafaelBlackwood/HousingPricePredictionMini.git
cd HousingPricePredictionMini

# create and activate a virtual env
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS and Linux
source .venv/bin/activate

# install project dependencies if requirements.txt exists
# otherwise install common libs used in tabular ML
pip install -r requirements.txt || pip install numpy pandas scikit-learn matplotlib jupyter
```

---

## Usage

There are two common ways to run this project. Pick the one that matches the code inside `pythonProject`.

### Option 1. Run a script

If the folder contains an entry script like `main.py` or `train.py` you can run:

```bash
python pythonProject/main.py
# or
python pythonProject/train.py
```

**Typical script flow**

1. Load dataset from a CSV path
2. Clean and encode features
3. Split into train and test
4. Train a model such as `LinearRegression`, `RandomForestRegressor` or `XGBoost`
5. Print metrics like MAE and RMSE
6. Save the trained model to `artifacts/model.pkl`

You can usually pass arguments like dataset path or model type

```bash
python pythonProject/train.py --data data/housing.csv --model random_forest --test-size 0.2 --seed 42
```

### Option 2. Work in a notebook

If this project includes a notebook, launch Jupyter and open it

```bash
jupyter notebook
```

Run the cells top to bottom to reproduce the pipeline and plots.

---

## Configuration

Create a `.env` file or edit a `config.yaml` if present. Common options

```
DATA_PATH=data/housing.csv
TARGET=SalePrice
TEST_SIZE=0.2
RANDOM_STATE=42
MODEL=random_forest
N_JOBS=-1
```

If there is no config file, pass these values as command line flags.

---

## Experiments and results

### Baseline

- Numeric imputation with median and simple encoding
- Linear Regression or Ridge as a quick baseline

### Tree based models

- Random Forest and Gradient Boosting often reduce MAE and RMSE on tabular data

### Reporting

- Final numbers, plots, and short discussion are in `report.docx`
- A summary deck is in `HousePricingPrediction-Rafael A..pptx`

These documents live at the root of the repository. ([GitHub][1])

---

## Troubleshooting

- If imports fail, make sure the virtual environment is active and all dependencies are installed
- If the dataset path is wrong, provide a full path with `--data /full/path/to/your.csv`
- If the project uses a different Python version, create the venv with that version
- If you get encoding errors, save the CSV in UTF-8 or pass `encoding="utf-8"` when reading

---

## Roadmap

- Add a `requirements.txt` or `pyproject.toml`
- Add a clear entry point script with `argparse`
- Add model saving and loading with `joblib`
- Add experiment tracking with a simple CSV log
- Add unit tests for data transforms and metrics
- Add CI workflow to run tests on push

---

## License

No license file is included yet. If you plan to share or reuse this work, add a `LICENSE` file to make the terms clear. ([GitHub][1])

---

## Acknowledgments

Built as a mini learning project to practice a clean ML workflow with tabular data.

[1]: https://github.com/RafaelBlackwood/HousingPricePredictionMini "GitHub - RafaelBlackwood/HousingPricePredictionMini"
