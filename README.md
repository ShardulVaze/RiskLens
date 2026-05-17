---
title: RiskLens
colorFrom: purple
colorTo: yellow
sdk: docker
pinned: false
license: mit
---

# RiskLens

A machine learning system that predicts the probability of loan default for credit applicants.

## Overview

RiskLens is an end-to-end ML pipeline that ingests raw loan application data, engineers features using PostgreSQL, trains an XGBoost classifier, and serves predictions through a Flask web interface.

## Architecture

Raw CSVs → PostgreSQL → SQL Feature Engineering → XGBoost Model → Flask API → Web UI

## Model Performance

- ROC-AUC: 0.748
- Algorithm: XGBoost with SMOTE for class imbalance
- Features: 49 engineered features
- Dataset: Home Credit Default Risk (307,511 applicants)

## Tech Stack

- Backend: Python, Flask
- Database: PostgreSQL
- ML: XGBoost, Scikit-learn, Imbalanced-learn
- Deployment: Docker, Hugging Face Spaces

## Project Structure

RiskLens/
├── app.py                  # Flask web application
├── database/               # DB configuration
├── ingestion/              # Data ingestion scripts
├── preprocessing/          # Feature engineering
├── ml_model/               # Model training scripts
├── templates/              # HTML templates
├── static/                 # CSS and JS assets
├── test_db.py              # Database tests
└── test_model.py           # Model tests

## Setup

1. Clone the repo
2. Create a .env file with your database credentials
3. Install dependencies: pip install -r requirements.txt
4. Run the app: python app.py

## Environment Variables

DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_PORT=5432
DB_NAME=your_db_name

## Features Used

- Loan amount ratios (credit/income, EMI/credit)
- Applicant demographics (age, employment years, family size)
- External credit bureau scores (EXT_SOURCE_1/2/3)
- Previous loan history (rejection rate, approval count)
- Regional and contact information flags