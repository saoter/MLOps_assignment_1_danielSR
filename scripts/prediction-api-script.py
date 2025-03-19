#!/usr/bin/env python3
# scripts/prediction_api.py

import os
import json
import pickle
import requests
import pandas as pd
from datetime import datetime

# API endpoint
API_URL = "http://130.225.39.127:8000/new_penguin/"

def fetch_penguin_data():
    """Fetch new penguin data from the API."""
    try:
        response = requests.get(API_URL)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Parse the JSON response
        penguin_data = response.json()
        print("Fetched penguin data:", penguin_data)
        
        return penguin_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def load_model():
    """Load the trained model from disk."""
    model_path = "models/penguin_classifier.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def make_prediction(model, penguin_data):
    """Make a prediction based on the penguin data."""
    # Create a DataFrame with the required features
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    # Check if all required features are present
    if not all(feature in penguin_data for feature in features):
        print("Error: Missing required features in penguin data")
        return None
    
    # Create a dataframe with the penguin data
    df = pd.DataFrame([penguin_data])
    
    # Make the prediction
    try:
        species_prediction = model.predict(df[features])[0]
        probability = max(model.predict_proba(df[features])[0]) * 100
        
        result = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'features': penguin_data,
            'prediction': species_prediction,
            'confidence': f"{probability:.2f}%"
        }
        
        return result
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

def save_prediction(prediction):
    """Save the prediction to a JSON file."""
    # Create predictions directory if it doesn't exist
    os.makedirs('data/predictions', exist_ok=True)
    
    # Generate filename with current date
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f"data/predictions/prediction_{date_str}.json"
    
    # Save prediction to file
    with open(filename, 'w') as f:
        json.dump(prediction, f, indent=4)
    
    print(f"Prediction saved to {filename}")
    
    # Also save to a latest prediction file for the website
    with open('data/predictions/latest_prediction.json', 'w') as f:
        json.dump(prediction, f, indent=4)

def update_website():
    """Update the website with the latest prediction."""
    try:
        # Load the latest prediction
        with open('data/predictions/latest_prediction.json', 'r') as f:
            prediction = json.load(f)
        
        # Load all historical predictions
        predictions_dir = 'data/predictions'
        all_predictions = []
        
        for filename in os.listdir(predictions_dir):
            if filename.startswith('prediction_') and filename.endswith('.json'):
                file_path = os.path.join(predictions_dir, filename)
                with open(file_path, 'r') as f:
                    all_predictions.append(json.load(f))
        
        # Sort predictions by date (newest first)
        all_predictions.sort(key=lambda x: x['date'], reverse=True)
        
        # Generate HTML content for index.html
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Penguin Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; }}
        .prediction-card {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">Penguin Species Classifier</h1>
        
        <div class="card prediction-card">
            <div class="card-header bg-primary text-white">
                <h3>Latest Prediction - {prediction['date']}</h3>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h4>Features:</h4>
                        <ul class="list-group">
                            <li class="list-group-item">Bill Length: {prediction['features']['bill_length_mm']} mm</li>
                            <li class="list-group-item">Bill Depth: {prediction['features']['bill_depth_mm']} mm</li>
                            <li class="list-group-item">Flipper Length: {prediction['features']['flipper_length_mm']} mm</li>
                            <li class="list-group-item">Body Mass: {prediction['features']['body_mass_g']} g</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h4>Prediction:</h4>
                        <div class="alert {"alert-success" if prediction['prediction'] == 'Adelie' else "alert-secondary"}">
                            <h5>{prediction['prediction']} Penguin</h5>
                            <p>Confidence: {prediction['confidence']}</p>
                            {"<p class='text-success fw-bold'>This might be one of our Penguins of Madagascar!</p>" if prediction['prediction'] == 'Adelie' else ""}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <h2>Prediction History</h2>
        <div class="table-responsive">
            <table class="table table-striped">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Time</th>
                        <th>Bill Length (mm)</th>
                        <th>Bill Depth (mm)</th>
                        <th>Flipper Length (mm)</th>
                        <th>Body Mass (g)</th>
                        <th>Species</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        # Add prediction history rows
        for p in all_predictions:
            html_content += f"""
                    <tr>
                        <td>{p['date']}</td>
                        <td>{p['time']}</td>
                        <td>{p['features']['bill_length_mm']}</td>
                        <td>{p['features']['bill_depth_mm']}</td>
                        <td>{p['features']['flipper_length_mm']}</td>
                        <td>{p['features']['body_mass_g']}</td>
                        <td{"class='table-success'" if p['prediction'] == 'Adelie' else ""}>{p['prediction']}</td>
                        <td>{p['confidence']}</td>
                    </tr>
"""
        
        # Complete the HTML
        html_content += """
                </tbody>
            </table>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        
        # Write the updated HTML to index.html
        with open('index.html', 'w') as f:
            f.write(html_content)
        
        print("Website updated successfully")
    
    except Exception as e:
        print(f"Error updating website: {e}")

def main():
    """Main function to fetch data, make predictions, and update the website."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting penguin prediction service...")
    
    # Fetch penguin data from API
    penguin_data = fetch_penguin_data()
    if not penguin_data:
        print("Failed to fetch penguin data. Exiting.")
        return
    
    # Load the trained model
    model = load_model()
    if not model:
        print("Failed to load model. Exiting.")
        return
    
    # Make prediction
    prediction = make_prediction(model, penguin_data)
    if not prediction:
        print("Failed to make prediction. Exiting.")
        return
    
    # Save prediction
    save_prediction(prediction)
    
    # Update website
    update_website()
    
    # Log information about the prediction
    print(f"Prediction completed: {prediction['prediction']} with {prediction['confidence']} confidence")
    if prediction['prediction'] == 'Adelie':
        print("This might be one of our Penguins of Madagascar!")

if __name__ == "__main__":
    main()
