# Housing Price Prediction

This project trains and compares regression models for predicting house prices from a small tabular dataset. It includes data preparation, feature engineering, model evaluation, saved artifacts, and reproducible plots.

## Project Structure

```text
.
|-- pythonProject/
|   |-- housing_price_dataset.csv
|   |-- project_ai.py
|   `-- plots/
|-- HousePricingPrediction-Rafael A..pptx
|-- Project.docx
|-- report.docx
|-- requirements.txt
`-- README.md
```

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Run the Training Pipeline

From the repository root:

```bash
python pythonProject/project_ai.py
```

The script uses `pythonProject/housing_price_dataset.csv` by default. You can override the input and output paths:

```bash
python pythonProject/project_ai.py --data-path pythonProject/housing_price_dataset.csv --reference-year 2024
```

## What the Script Produces

The training run:

- validates the dataset columns
- creates `TotalRooms` and `PropertyAge` features
- compares a baseline, linear models, and tree-based regressors
- evaluates models with MAE, MSE, RMSE, and R2
- saves the best trained pipeline to `pythonProject/artifacts/`
- writes metrics and test predictions to CSV files
- regenerates plots in `pythonProject/plots/`

Generated plots include:

- `model_performance.png`
- `mean_absolute_error.png`
- `mean_squared_error.png`
- `root_mean_squared_error.png`
- `actual_vs_predicted.png`
- `residuals.png`
- `feature_importance.png` when the selected model supports it

## Notes

The model artifact files are generated outputs and are ignored by git. The plot images are kept in the repository because they are useful for reports and quick review.
