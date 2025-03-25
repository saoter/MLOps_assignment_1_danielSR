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
import seaborn as sns
from datetime import datetime
from pathlib import Path
import schedule
import time

# Set matplotlib backend to Agg for server environments
matplotlib.use('Agg')

# API endpoint for new penguin data
API_URL = "http://130.225.39.127:8000/new_penguin/"

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
    """Load the trained models."""
    print("Loading trained models...")
    species_model_path = "models/best_model.pkl"
    if not os.path.exists(species_model_path):
        print(f"Error: Species model file not found at {species_model_path}")
        return None
    species_model = joblib.load(species_model_path)
    try:
        with open("models/model_metrics.json", "r") as f:
            model_info = json.load(f)
        print(f"Using {model_info['model_name']} model for species classification")
    except:
        model_info = {"model_name": "Unknown"}
        print("Model information not found, using loaded model without additional info")
    return species_model, model_info

def make_prediction(species_model, model_info, penguin_data):
    """Make prediction for the new penguin data."""
    print("Making prediction...")
    df = pd.DataFrame([penguin_data])
    required_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    for feature in required_features:
        if feature not in df.columns:
            print(f"Warning: Missing required feature {feature}. Using default value.")
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
        "note": f"This is likely one of the Penguins of Madagascar!" if is_adelie else "Not one of the Penguins of Madagascar"
    }
    print(f"Prediction: {species_prediction}")
    print(f"Is Adelie: {is_adelie}")
    return prediction_result

def create_visualization(prediction_result):
    """Create a visualization of the prediction result."""
    print("Creating visualization...")
    penguin_data = prediction_result["penguin_data"]
    species = prediction_result["predicted_species"]
    is_adelie = prediction_result["is_adelie"]
    fig, ax1 = plt.subplots(figsize=(8, 6))
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm/10', 'body_mass_g/100']
    values = [
        penguin_data['bill_length_mm'],
        penguin_data['bill_depth_mm'],
        penguin_data['flipper_length_mm'] / 10,  # Scale down for better visualization
        penguin_data['body_mass_g'] / 100        # Scale down for better visualization
    ]
    colors = ['#1e3d59' if not is_adelie else '#ff6e40'] * 4
    ax1.bar(features, values, color=colors)
    ax1.set_title("Penguin Measurements", fontsize=14, weight='bold')
    ax1.set_ylabel("Value")
    for i, v in enumerate(values):
        ax1.text(i, v + 0.5, f"{v:.1f}", ha='center')
    plt.tight_layout()
    try:
        Path("data/predictions").mkdir(parents=True, exist_ok=True)
        plt.savefig("data/predictions/latest_visualization.png")
        print("Visualization saved successfully")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    plt.close()

def save_prediction_results(prediction_result):
    """Save prediction results to a file."""
    print("Saving prediction results...")
    Path("data/predictions").mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    prediction_path = f"data/predictions/{today}.json"
    with open(prediction_path, "w") as f:
        json.dump(prediction_result, f, indent=2)
    with open("data/predictions/latest.json", "w") as f:
        json.dump(prediction_result, f, indent=2)
    print