#!/usr/bin/env python3
# scripts/model_training.py

import os
import sqlite3
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def load_data_from_db():
    """Load data from SQLite database."""
    conn = sqlite3.connect('data/penguins.db')
    
    # Join penguins and islands tables
    query = '''
    SELECT p.animal_id, p.species, p.bill_length_mm, p.bill_depth_mm, 
           p.flipper_length_mm, p.body_mass_g, p.sex, i.name as island
    FROM penguins p
    JOIN islands i ON p.island_id = i.island_id
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def perform_feature_analysis(df):
    """Analyze features and their importance for classification."""
    # Create directory for plots
    os.makedirs('data/plots', exist_ok=True)
    
    # Separating features and target
    X = df[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
    y = df['species']
    
    # Feature correlation analysis
    plt.figure(figsize=(10, 8))
    correlation = X.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('data/plots/feature_correlation.png')
    
    # Feature distributions by species
    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(X.columns):
        plt.subplot(2, 2, i+1)
        sns.boxplot(x='species', y=feature, data=df)
        plt.title(f'{feature} by Species')
    plt.tight_layout()
    plt.savefig('data/plots/feature_distributions.png')
    
    # Feature importance using SelectKBest (ANOVA F-test)
    selector = SelectKBest(f_classif, k=4)
    selector.fit(X, y)
    feature_scores = pd.DataFrame({
        'Feature': X.columns,
        'F-Score': selector.scores_,
        'p-value': selector.pvalues_
    }).sort_values('F-Score', ascending=False)
    
    return feature_scores

def train_and_evaluate_models(df):
    """Train different models and evaluate their performance."""
    # Prepare data
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    X = df[features]
    y = df['species']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Define models to try
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVC': SVC(random_state=42),
        'KNN': KNeighborsClassifier()
    }
    
    # Define param grids for each model
    param_grids = {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        },
        'SVC': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    }
    
    # Define the pipeline with scaling
    results = {}
    best_model = None
    best_score = 0
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            pipeline, 
            {f'model__{param}': value for param, value in param_grids[model_name].items()},
            cv=5, 
            scoring='accuracy',
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Evaluate the best model from grid search
        y_pred = grid_search.predict(X_test)
        accuracy = grid_search.score(X_test, y_test)
        
        print(f"{model_name} Best Parameters: {grid_search.best_params_}")
        print(f"{model_name} Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save the confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=grid_search.classes_,
                    yticklabels=grid_search.classes_)
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True Species')
        plt.xlabel('Predicted Species')
        plt.tight_layout()
        plt.savefig(f'data/plots/{model_name}_confusion_matrix.png')
        
        # Store results
        results[model_name] = {
            'model': grid_search,
            'accuracy': accuracy,
            'best_params': grid_search.best_params_
        }
        
        # Track the best model
        if accuracy > best_score:
            best_score = accuracy
            best_model = model_name
    
    return results, best_model

def save_model(model, filename):
    """Save the trained model to disk."""
    os.makedirs('models', exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

def main():
    """Main function to train and save the model."""
    print("Loading data from database...")
    df = load_data_from_db()
    
    print("\nPerforming feature analysis...")
    feature_scores = perform_feature_analysis(df)
    print("Feature importance (ANOVA F-test):")
    print(feature_scores)
    
    print("\nTraining and evaluating models...")
    results, best_model = train_and_evaluate_models(df)
    
    print(f"\nBest model: {best_model} with accuracy {results[best_model]['accuracy']:.4f}")
    
    # Save the best model
    best_model_filename = f"models/penguin_classifier.pkl"
    save_model(results[best_model]['model'], best_model_filename)
    
    # Save feature importance analysis
    feature_scores.to_csv('data/feature_importance.csv', index=False)
    
    # Save a summary of model results
    summary = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results.keys()],
        'Best_Parameters': [str(results[model]['best_params']) for model in results.keys()]
    })
    summary.to_csv('data/model_comparison.csv', index=False)
    print("\nModel training complete. Results saved to data directory.")

if __name__ == "__main__":
    main()
