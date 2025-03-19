#!/usr/bin/env python3
# scripts/data_processing.py

import os
import sqlite3
import pandas as pd
import seaborn as sns
import numpy as np

def create_database_structure(conn):
    """Create the database tables according to the schema."""
    cursor = conn.cursor()
    
    # Create ISLANDS table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS islands (
        island_id INTEGER PRIMARY KEY,
        name TEXT
    )
    ''')
    
    # Create PENGUINS table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS penguins (
        animal_id INTEGER PRIMARY KEY,
        species TEXT,
        bill_length_mm REAL,
        bill_depth_mm REAL,
        flipper_length_mm REAL,
        body_mass_g REAL,
        sex TEXT,
        island_id INTEGER,
        FOREIGN KEY (island_id) REFERENCES islands (island_id)
    )
    ''')
    
    conn.commit()

def populate_database(conn, penguins_df):
    """Populate the database with penguin data."""
    # Extract unique islands and create a mapping
    islands = penguins_df['island'].unique()
    island_mapping = {island: i+1 for i, island in enumerate(islands)}
    
    # Insert islands data
    cursor = conn.cursor()
    for island, island_id in island_mapping.items():
        cursor.execute('INSERT INTO islands (island_id, name) VALUES (?, ?)',
                      (island_id, island))
    
    # Insert penguins data
    for i, row in penguins_df.iterrows():
        cursor.execute('''
        INSERT INTO penguins (
            animal_id, species, bill_length_mm, bill_depth_mm, 
            flipper_length_mm, body_mass_g, sex, island_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            i+1,
            row['species'],
            row['bill_length_mm'],
            row['bill_depth_mm'],
            row['flipper_length_mm'],
            row['body_mass_g'],
            row['sex'],
            island_mapping[row['island']]
        ))
    
    conn.commit()

def main():
    """Main function to process the data and create the database."""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Loading penguins dataset...")
    # Load the penguins dataset
    penguins = sns.load_dataset("penguins").dropna()
    
    # Save the raw data as CSV for reference
    penguins.to_csv('data/penguins_raw.csv', index=False)
    print(f"Saved raw data to data/penguins_raw.csv ({len(penguins)} rows)")
    
    # Display dataset info
    print("\nDataset Information:")
    print(f"Number of samples: {len(penguins)}")
    print(f"Features: {', '.join(penguins.columns)}")
    print(f"Species distribution: {penguins['species'].value_counts().to_dict()}")
    
    # Connect to SQLite database
    db_path = 'data/penguins.db'
    conn = sqlite3.connect(db_path)
    
    # Create and populate the database
    create_database_structure(conn)
    populate_database(conn, penguins)
    
    # Verify the data was inserted correctly
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM penguins")
    penguin_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM islands")
    island_count = cursor.fetchone()[0]
    
    print(f"\nDatabase created at {db_path}")
    print(f"Inserted {penguin_count} penguins across {island_count} islands")
    
    # Close connection
    conn.close()

if __name__ == "__main__":
    main()
