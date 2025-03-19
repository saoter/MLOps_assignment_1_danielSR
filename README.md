# Penguin Species Classifier

A machine learning pipeline that identifies penguin species based on physical measurements. Created for the MLOps 2025 course assignment.

## Project Overview

This project aims to classify penguin species based on physical measurements to help identify Adelie penguins (like Skipper, Private, Rico, and Kowalski from the "Penguins of Madagascar") spotted in New York City.

The pipeline:
1. Processes penguin data into a SQL database
2. Trains a machine learning model to classify penguin species
3. Automatically fetches new penguin data daily from an API
4. Makes predictions and displays results on GitHub Pages

## Technical Implementation

### Data Processing
- Dataset: Palmer Penguins dataset from the Seaborn library
- Database: SQLite with tables for penguins and islands
- Processing script: `scripts/data_processing.py`

### Model Training
- Features: bill length, bill depth, flipper length, body mass
- Target: Penguin species classification
- Algorithms evaluated: [list algorithms tested]
- Best performing model: [model name] with [accuracy]
- Model training script: `scripts/model_training.py`

### Automated Prediction Pipeline
- API endpoint: http://130.225.39.127:8000/new_penguin/
- Schedule: Daily at 7:30 AM
- GitHub Actions workflow: `.github/workflows/daily_prediction.yml`
- Results: Published to GitHub Pages at [your-github-pages-url]

## How to Run

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/penguin-species-classifier.git
cd penguin-species-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Processing
```bash
python scripts/data_processing.py
```

### Model Training
```bash
python scripts/model_training.py
```

### Manual Prediction
```bash
python scripts/prediction_api.py
```

## Results

The latest penguin species predictions are available at [GitHub Pages URL].

## Technologies Used
- Python 3.10+
- Pandas for data manipulation
- Scikit-learn for model training and evaluation
- SQLite for data storage
- GitHub Actions for CI/CD and scheduled tasks
- GitHub Pages for displaying results
