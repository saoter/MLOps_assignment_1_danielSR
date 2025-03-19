#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified model training script that only uses features available in the API.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
from pathlib import Path

def load_data_from_db():
    """Load penguin data from SQLite database."""
    print("Loading data from SQLite database...")
    
    conn = sqlite3.connect("data/penguins.db")
    try:
        query = """
        SELECT p.species, p.bill_length_mm, p.bill_depth_mm, p.flipper_length_mm, p.body_mass_g
        FROM PENGUINS p
        """
        
        df = pd.read_sql(query, conn)
        print(f"Loaded {len(df)} records from database.")
        return df
    
    finally:
        conn.close()

def train_models_with_api_features():
    """
    Train models using only features available in the API.
    """
    # Load data
    df = load_data_from_db()
    
    # Define features available in the API
    api_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    
    # Prepare data
    X = df[api_features]
    y = df['species']
    
    # Encode target labels for XGBoost
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Split encoded target as well (for XGBoost)
    _, _, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Testing data: {X_test.shape[0]} samples")
    
    # Create preprocessing pipeline
    preprocessor = StandardScaler()
    
    # Define models
    models = {
        'RandomForest': {
            'pipeline': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20]
            },
            'needs_encoded_labels': False
        },
        'XGBoost': {
            'pipeline': Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', xgb.XGBClassifier(objective='multi:softprob', random_state=42))
            ]),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 5, 7]
            },
            'needs_encoded_labels': True
        }
    }
    
    best_accuracy = 0
    best_model_name = None
    best_model = None
    model_metrics = {}
    
    # Train each model
    for model_name, model_info in models.items():
        print(f"\nTraining {model_name}...")
        
        # Select appropriate target variable
        train_y = y_train_encoded if model_info['needs_encoded_labels'] else y_train
        test_y = y_test_encoded if model_info['needs_encoded_labels'] else y_test
        
        # Train with grid search
        grid_search = GridSearchCV(
            model_info['pipeline'],
            model_info['params'],
            cv=5,
            scoring='accuracy'
        )
        
        grid_search.fit(X_train, train_y)
        
        # Get the best model
        model = grid_search.best_estimator_
        
        # Make predictions
        if model_info['needs_encoded_labels']:
            # For XGBoost
            y_pred_encoded = model.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
            
            # Store label encoder with the model
            setattr(model, 'label_encoder', label_encoder)
        else:
            # For RandomForest
            y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        print(f"{model_name} Best Parameters: {grid_search.best_params_}")
        print(f"{model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store metrics
        model_metrics[model_name] = {
            'accuracy': float(accuracy),
            'best_params': grid_search.best_params_,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Update best model if better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_model = model
    
    # Save the best model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/api_model.pkl")
    
    # Save model info
    model_info = {
        'model_name': best_model_name,
        'metrics': model_metrics,
        'features_used': api_features,
        'target_classes': list(label_encoder.classes_)
    }
    
    with open("models/api_model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nBest model ({best_model_name}) saved to models/api_model.pkl")
    print(f"Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    train_models_with_api_features()