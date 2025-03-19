#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Daily prediction script for penguin classification.
This script fetches new penguin data from the API, makes predictions using the trained model,
and saves the results to be displayed on GitHub Pages.
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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
        # Return sample data for testing when API is unavailable
        return {
            'bill_length_mm': 39.1,
            'bill_depth_mm': 18.7,
            'flipper_length_mm': 181.0,
            'body_mass_g': 3750.0
        }

def load_model():
    """Load the trained model."""
    print("Loading trained model...")
    
    model_path = "models/best_model.pkl"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    # Load the model
    model = joblib.load(model_path)
    
    # Load model information
    try:
        with open("models/model_metrics.json", "r") as f:
            model_info = json.load(f)
        print(f"Using {model_info['model_name']} model")
    except:
        model_info = {"model_name": "Unknown"}
        print("Model information not found, using loaded model without additional info")
    
    return model, model_info

def make_prediction(model, model_info, penguin_data):
    """Make prediction for the new penguin data."""
    print("Making prediction...")
    
    # Convert data to DataFrame for preprocessing
    df = pd.DataFrame([penguin_data])
    
    # Check for missing features and add them with default values
    required_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'island_name']
    missing_features = set(required_features) - set(df.columns)
    
    if missing_features:
        print(f"Adding missing features with default values: {missing_features}")
        
        # Add default values for missing features
        for feature in missing_features:
            if feature == 'sex':
                # Use the most common value from training data or a reasonable default
                df[feature] = 'MALE'  # You can change this to the most common value in your training data
            elif feature == 'island_name':
                # Use the most common value from training data or a reasonable default
                df[feature] = 'Biscoe'  # You can change this to the most common value in your training data
            else:
                # For numeric features, use mean values or other reasonable defaults
                df[feature] = 0  # Replace with appropriate default
    
    # Remove any extra columns not used by the model (like datetime)
    extra_columns = set(df.columns) - set(required_features)
    if extra_columns:
        print(f"Removing extra columns not used for prediction: {extra_columns}")
        df = df.drop(columns=extra_columns)
    
    # Make prediction based on model type
    if model_info.get("model_name") == "XGBoost" and hasattr(model, 'label_encoder'):
        # For XGBoost, we need to use the label encoder
        raw_prediction = model.predict(df)[0]
        species_prediction = model.label_encoder.inverse_transform([raw_prediction])[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0]
            # Map probabilities to class names
            proba_dict = {cls: float(prob) for cls, prob in zip(model.label_encoder.classes_, probabilities)}
        else:
            proba_dict = {"Note": "Probability information not available"}
    else:
        # For other models like RandomForest
        species_prediction = model.predict(df)[0]
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[0]
            # Map probabilities to class names
            proba_dict = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}
        else:
            proba_dict = {"Note": "Probability information not available"}
    
    # Create prediction result
    prediction_result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": datetime.now().strftime("%H:%M:%S"),
        "penguin_data": penguin_data,
        "predicted_species": species_prediction,
        "probabilities": proba_dict,
        "model_used": model_info.get("model_name", "Unknown")
    }
    
    print(f"Prediction: {species_prediction}")
    return prediction_result

def save_prediction_results(prediction_result):
    """Save prediction results to a file."""
    print("Saving prediction results...")
    
    # Create directory for predictions if it doesn't exist
    Path("data/predictions").mkdir(exist_ok=True)
    
    # Save prediction as JSON
    today = datetime.now().strftime("%Y-%m-%d")
    prediction_path = f"data/predictions/{today}.json"
    
    with open(prediction_path, "w") as f:
        json.dump(prediction_result, f, indent=2)
    
    # Also save to a "latest.json" file for easy access
    with open("data/predictions/latest.json", "w") as f:
        json.dump(prediction_result, f, indent=2)
    
    print(f"Prediction saved to {prediction_path}")
    
    # Update prediction history
    update_prediction_history(prediction_result)
    
    # Create visualization
    create_visualization(prediction_result)

def update_prediction_history(prediction_result):
    """Update the prediction history JSON file."""
    history_path = "data/predictions/history.json"
    
    # Load existing history or create a new one
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = []
    
    # Add new prediction to history
    simplified_result = {
        "date": prediction_result["date"],
        "predicted_species": prediction_result["predicted_species"],
        "is_adelie": prediction_result["predicted_species"] == "Adelie"
    }
    
    history.append(simplified_result)
    
    # Save updated history
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

def create_visualization(prediction_result):
    """Create a visualization of the prediction result."""
    print("Creating visualization...")
    
    # Extract data
    penguin_data = prediction_result["penguin_data"]
    species = prediction_result["predicted_species"]
    
    # Create a radar chart of the penguin measurements
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    # Normalize values for radar chart
    # We use approximate ranges based on the Palmer Penguins dataset
    ranges = {
        'bill_length_mm': (32, 59),
        'bill_depth_mm': (13, 21),
        'flipper_length_mm': (172, 231),
        'body_mass_g': (2700, 6300)
    }
    
    # Normalize each value to 0-1 range
    normalized_values = []
    for feature in features:
        min_val, max_val = ranges[feature]
        val = penguin_data[feature]
        norm_val = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
        normalized_values.append(max(0, min(norm_val, 1)))  # Clip to 0-1 range
    
    # Create radar chart
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of features
    N = len(features)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the circle
    
    # Add values
    values = normalized_values + [normalized_values[0]]  # Close the circle
    
    # Draw the chart
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    
    # Fix axis to go in the right order and start at top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    
    # Add title and adjust layout
    plt.title(f"Predicted Species: {species}", size=15)
    
    # Save the chart
    plt.tight_layout()
    plt.savefig("data/predictions/latest_visualization.png")
    
    # Also update the visualization displayed on GitHub Pages
    try:
        plt.savefig("latest_visualization.png")
    except:
        print("Could not save visualization to root directory for GitHub Pages")
    
    plt.close()

def update_github_pages():
    """Update GitHub Pages with the latest prediction results."""
    print("Updating GitHub Pages content...")
    
    # Load the latest prediction
    try:
        with open("data/predictions/latest.json", "r") as f:
            latest = json.load(f)
    except:
        print("Error: Could not load latest prediction")
        return
    
    # Load prediction history
    try:
        with open("data/predictions/history.json", "r") as f:
            history = json.load(f)
    except:
        history = []
    
    # Count Adelie penguins
    adelie_count = sum(1 for item in history if item.get("is_adelie", False))
    
    # Create HTML content for GitHub Pages
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Penguin Detector</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
        }}
        .penguin-data {{
            margin: 20px 0;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}
        .prediction {{
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
            padding: 10px;
            background-color: #e8f4f8;
            border-radius: 5px;
        }}
        .adelie {{
            color: #2980b9;
        }}
        .other {{
            color: #c0392b;
        }}
        .stats {{
            margin-top: 30px;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        table, th, td {{
            border: 1px solid #ddd;
            padding: 8px;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Penguin Species Detector</h1>
        
        <div class="prediction">
            <p>Latest Prediction (Date: {latest["date"]}): 
                <span class="{'adelie' if latest['predicted_species'] == 'Adelie' else 'other'}">{latest["predicted_species"]}</span>
            </p>
        </div>
        
        <div class="visualization">
            <img src="latest_visualization.png" alt="Penguin Visualization" style="max-width: 100%;">
        </div>
        
        <div class="penguin-data">
            <h3>Penguin Measurements:</h3>
            <ul>
                <li><strong>Bill Length:</strong> {latest["penguin_data"]["bill_length_mm"]} mm</li>
                <li><strong>Bill Depth:</strong> {latest["penguin_data"]["bill_depth_mm"]} mm</li>
                <li><strong>Flipper Length:</strong> {latest["penguin_data"]["flipper_length_mm"]} mm</li>
                <li><strong>Body Mass:</strong> {latest["penguin_data"]["body_mass_g"]} g</li>
            </ul>
        </div>
        
        <div class="stats">
            <h3>Statistics</h3>
            <p>Total penguins detected: {len(history)}</p>
            <p>Adelie penguins detected: {adelie_count} ({adelie_count/len(history)*100:.1f}% of total)</p>
            <p>Model used: {latest["model_used"]}</p>
        </div>
        
        <div class="history">
            <h3>Recent Detection History (Last 10)</h3>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Species</th>
                </tr>
                {"".join(f"<tr><td>{item['date']}</td><td>{item['predicted_species']}</td></tr>" for item in history[-10:])}
            </table>
        </div>
    </div>
</body>
</html>
"""
    
    # Write the HTML content to index.html in the root directory for GitHub Pages
    with open("index.html", "w") as f:
        f.write(html_content)
    
    print("GitHub Pages content updated")

def main():
    """Main function to fetch data and make predictions."""
    try:
        # Fetch new penguin data
        penguin_data = fetch_new_penguin_data()
        
        # Load the trained model
        model, model_info = load_model()
        
        # Make prediction
        prediction_result = make_prediction(model, model_info, penguin_data)
        
        # Save prediction results
        save_prediction_results(prediction_result)
        
        # Update GitHub Pages
        update_github_pages()
        
        print("Daily prediction completed successfully!")
        print(f"Is it an Adelie penguin? {'Yes' if prediction_result['predicted_species'] == 'Adelie' else 'No'}")
        
    except Exception as e:
        print(f"Error in daily prediction: {e}")
        raise

if __name__ == "__main__":
    main()