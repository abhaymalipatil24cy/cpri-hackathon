# CPRI Hackathon — Data Science & ML Pipeline

## Project Overview
This repository contains the end-to-end data-science pipeline for the CPRI State-Level Hackathon.
It performs data quality auditing, missing value imputation, operating regime clustering, cross-sensor physical consistency modeling, hybrid validity detection, reference parameter regression, uncertainty scoring, and attention prioritization.

## Repository Structure
``ncpri-hackathon/
├── data/
│   ├── training_data.csv
│   └── test_data.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest.py
│   ├── clean_impute.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── abnormality_detection.py
│   ├── regression_model.py
│   ├── summary_report.py
│   └── verify.py
├── tests/
│   └── test_cp0_environment.py
├── artifacts/
│   ├── data_quality/
│   ├── cleaning/
│   ├── regimes/
│   ├── validity/
│   ├── regression/
│   ├── experiments/
│   └── verification/
├── outputs/
│   ├── TeamName.csv
│   └── summary.json
├── run_pipeline.py
├── requirements.txt
└── README.md
``n
## Environment Setup
1. Python 3.14+ (or compatible 3.10+)
2. Create and activate a virtual environment:
   `ash
   uv venv .venv
   .venv\Scripts\activate
   ``n3. Install dependencies:
   `ash
   pip install -r requirements.txt
   ``n
## How to Run Tests
`ash
pytest tests/
``n
## How to Run Pipeline
`ash
python run_pipeline.py
``n