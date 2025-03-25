#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Scheduled prediction script for penguin classification.
This script fetches new penguin data from the API at regular intervals,
makes predictions using the trained model, and saves the results locally.
"""

import os
import json
import requests
import pandas as pd
import joblib
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

# Set matplotlib backend to Agg for server environments
matplotlib.use('Agg')

# API endpoint for new penguin data
API_URL = "http://130.225.39.127:8000/new_penguin/"

# Define the output directory
OUTPUT_DIR = Path("data/predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def fetch_new_penguin_data():
    """Fetch new penguin data from the API."""
    print(f"Fetching new penguin data from {API_URL}...")
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()  # Raise exception for HTTP errors
        data = response.json()
        print(f"Successfully fetched data: {data}")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def load_models():
    """Load the trained model."""
    print("Loading trained model...")
    species_model_path = "models/best_model.pkl"
    if not os.path.exists(species_model_path):
        print(f"Error: Model file not found at {species_model_path}")
        return None, None
    species_model = joblib.load(species_model_path)
    model_info_path = "models/model_metrics.json"
    if os.path.exists(model_info_path):
        with open(model_info_path, "r") as f:
            model_info = json.load(f)
        print(f"Using {model_info.get('model_name', 'Unknown')} model for species classification")
    else:
        model_info = {"model_name": "Unknown"}
        print("Model information not found, using loaded model without additional info")
    return species_model, model_info

def make_prediction(species_model, model_info, penguin_data):
    """Make prediction for the new penguin data."""
    print("Making prediction...")
    df = pd.DataFrame([penguin_data])
    required_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    missing_features = [feature for feature in required_features if feature not in df.columns]
    if missing_features:
        print(f"Warning: Missing required features {missing_features}. Using default value 0.")
        for feature in missing_features:
            df[feature] = 0  # Default value
    extra_columns = set(df.columns) - set(required_features)
    if extra_columns:
        print(f"Removing extra columns not used for prediction: {extra_columns}")
        df = df.drop(columns=extra_columns)
    species_prediction = species_model.predict(df)[0]
    if hasattr(species_model, 'predict_proba'):
        probabilities = species_model.predict_proba(df)[0]
        proba_dict = {cls: float(prob) for cls, prob in zip(species_model.classes_, probabilities)}
    else:
        proba_dict = {"Note": "Probability information not available"}
    is_adelie = (species_prediction == 'Adelie')
    prediction_result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "penguin_data": penguin_data,
        "predicted_species": species_prediction,
        "species_probabilities": proba_dict,
        "is_adelie": is_adelie,
        "model_used": model_info.get("model_name", "Unknown"),
        "note": "This is likely one of the Penguins of Madagascar!" if is_adelie else "Not one of the Penguins of Madagascar"
    }
    print(f"Prediction: {species_prediction}")
    print(f"Is Adelie: {is_adelie}")
    return prediction_result

def create_visualization(prediction_result):
    """Create a visualization of the prediction result."""
    print("Creating visualization...")
    penguin_data = prediction_result["penguin_data"]
    is_adelie = prediction_result["is_adelie"]
    fig, ax1 = plt.subplots(figsize=(8, 6))
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    values = [
        penguin_data['bill_length_mm'],
        penguin_data['bill_depth_mm'],
        penguin_data['flipper_length_mm'],
        penguin_data['body_mass_g']
    ]
    colors = ['#ff6e40' if is_adelie else '#1e3d59'] * 4
    ax1.bar(features, values, color=colors)
    ax1.set_title("Penguin Measurements", fontsize=14, weight='bold')
    ax1.set_ylabel("Value")
    for i, v in enumerate(values):
        ax1.text(i, v + 0.5, f"{v:.1f}", ha='center')
    plt.tight_layout()
    visualization_path = OUTPUT_DIR / "latest_visualization.png"
    try:
        plt.savefig(visualization_path)
        print(f"Visualization saved successfully at {visualization_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    plt.close()

def save_prediction_results(prediction_result):
    """Save prediction results to files."""
    print("Saving prediction results...")
    today = datetime.now().strftime("%Y-%m-%d")
    prediction_path = OUTPUT_DIR / f"{today}.json"
    latest_path = OUTPUT_DIR / "latest.json"
    try:
        with open(prediction_path, "w") as f:
            json.dump(prediction_result, f, indent=2)
        with open(latest_path, "w") as f:
            json.dump(prediction_result, f, indent=2)
        print(f"Prediction results saved successfully at {prediction_path} and {latest_path}")
    except Exception as e:
        print(f"Error saving prediction results: {e}")

def generate_html_report(prediction_result):
    """Generate an HTML report from the prediction result."""
    env = Environment(loader=FileSystemLoader('/app/templates'))
    template = env.get_template('index.html.template')
    html_content = template.render(
        PREDICTION_DATE=prediction_result["date"],
        PREDICTION_TIME=prediction_result["time"],
        PREDICTED_SPECIES=prediction_result["predicted_species"],
        BILL_LENGTH=prediction_result["penguin_data"]["bill_length_mm"],
        BILL_DEPTH=prediction_result["penguin_data"]["bill_depth_mm"],
        FLIPPER_LENGTH=prediction_result["penguin_data"]["flipper_length_mm"],
        BODY_MASS=prediction_result["penguin_data"]["body_mass_g"],
        SEX=prediction_result["penguin_data"].get("sex", "Unknown"),
        PREDICTION_NOTE=prediction_result["note"]
    )
    output_path = OUTPUT_DIR / "index.html"
    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"HTML report generated at {output_path}")

def main():
    penguin_data = fetch_new_penguin_data()
    if not penguin_data:
        print("No data fetched. Exiting.")
        return
    species_model, model_info = load_models()
    if not species_model:
        print("Model loading failed. Exiting.")
        return
    prediction_result = make_prediction(species_model, model_info, penguin_data)
    create_visualization(prediction_result)
    save_prediction_results(prediction_result)
    print("Prediction process completed successfully.")

if __name__ == "__main__":
    main()
