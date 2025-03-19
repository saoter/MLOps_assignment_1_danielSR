#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Character matcher for Penguins of Madagascar.
This script identifies which specific penguin (Skipper, Kowalski, Rico, or Private)
a detected Adelie penguin is most likely to be.
"""

import os
import json
import numpy as np
from scipy.spatial.distance import euclidean
from pathlib import Path

# Configuration
CHARACTERS_FILE = "data/penguin_characters.json"
MEASUREMENT_KEYS = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Feature importance weights (these can be tuned based on which features better distinguish the characters)
FEATURE_WEIGHTS = {
    'bill_length_mm': 0.3,
    'bill_depth_mm': 0.2,
    'flipper_length_mm': 0.3,
    'body_mass_g': 0.2
}

def load_character_profiles():
    """Load character profiles from the JSON file."""
    try:
        with open(CHARACTERS_FILE, 'r') as f:
            data = json.load(f)
        return data['characters']
    except Exception as e:
        print(f"Error loading character profiles: {e}")
        # Return default profiles if file is missing
        return [
            {
                "name": "Skipper",
                "measurements": {"bill_length_mm": 39.5, "bill_depth_mm": 18.3, "flipper_length_mm": 187.0, "body_mass_g": 3800.0}
            },
            {
                "name": "Kowalski",
                "measurements": {"bill_length_mm": 41.2, "bill_depth_mm": 17.5, "flipper_length_mm": 196.0, "body_mass_g": 3750.0}
            },
            {
                "name": "Rico",
                "measurements": {"bill_length_mm": 38.1, "bill_depth_mm": 20.2, "flipper_length_mm": 181.0, "body_mass_g": 4100.0}
            },
            {
                "name": "Private",
                "measurements": {"bill_length_mm": 36.7, "bill_depth_mm": 16.8, "flipper_length_mm": 175.0, "body_mass_g": 3450.0}
            }
        ]

def normalize_measurements(characters, penguin_data):
    """
    Normalize measurements to ensure features are on the same scale.
    Returns normalized character measurements and normalized penguin data.
    """
    # Extract all measurements for normalization
    all_measurements = {}
    for key in MEASUREMENT_KEYS:
        all_measurements[key] = [char['measurements'][key] for char in characters] + [penguin_data[key]]
    
    # Calculate min and max for each measurement
    min_max = {}
    for key in MEASUREMENT_KEYS:
        min_max[key] = {
            'min': min(all_measurements[key]),
            'max': max(all_measurements[key])
        }
    
    # Normalize character measurements
    normalized_characters = []
    for char in characters:
        normalized_char = char.copy()
        normalized_char['normalized_measurements'] = {}
        for key in MEASUREMENT_KEYS:
            range_value = min_max[key]['max'] - min_max[key]['min']
            if range_value == 0:  # Avoid division by zero
                normalized_char['normalized_measurements'][key] = 0
            else:
                normalized_char['normalized_measurements'][key] = (
                    (char['measurements'][key] - min_max[key]['min']) / range_value
                )
        normalized_characters.append(normalized_char)
    
    # Normalize penguin data
    normalized_penguin = {}
    for key in MEASUREMENT_KEYS:
        range_value = min_max[key]['max'] - min_max[key]['min']
        if range_value == 0:  # Avoid division by zero
            normalized_penguin[key] = 0
        else:
            normalized_penguin[key] = (
                (penguin_data[key] - min_max[key]['min']) / range_value
            )
    
    return normalized_characters, normalized_penguin

def calculate_weighted_distance(normalized_char, normalized_penguin):
    """Calculate weighted Euclidean distance between character and penguin measurements."""
    weighted_distances = []
    
    for key in MEASUREMENT_KEYS:
        weighted_distance = (
            FEATURE_WEIGHTS[key] * 
            abs(normalized_char['normalized_measurements'][key] - normalized_penguin[key])
        )
        weighted_distances.append(weighted_distance)
    
    return sum(weighted_distances)

def calculate_similarity_score(distance):
    """Convert distance to a similarity score (0-100%)."""
    # Exponential decay function to convert distance to similarity
    # 0 distance = 100% similarity, larger distances approach 0%
    return 100 * np.exp(-3 * distance)

def match_character(penguin_data):
    """
    Match a detected penguin to one of the four Madagascar penguins.
    Returns a dictionary with character match information.
    """
    # Load character profiles
    characters = load_character_profiles()
    
    # Normalize measurements
    normalized_characters, normalized_penguin = normalize_measurements(characters, penguin_data)
    
    # Calculate distances and similarities
    character_matches = []
    for char in normalized_characters:
        distance = calculate_weighted_distance(char, normalized_penguin)
        similarity = calculate_similarity_score(distance)
        
        character_matches.append({
            "name": char["name"],
            "similarity": similarity,
            "profile": char.get("profile", ""),
            "role": char.get("role", ""),
            "catchphrase": char.get("catchphrase", ""),
            "skills": char.get("skills", []),
            "classified_info": char.get("classified_info", "")
        })
    
    # Sort by similarity (highest first)
    character_matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Calculate relative probabilities
    total_similarity = sum(char["similarity"] for char in character_matches)
    if total_similarity > 0:
        for char in character_matches:
            char["probability"] = char["similarity"] / total_similarity * 100
    
    # Create the result
    result = {
        "best_match": character_matches[0],
        "all_matches": character_matches,
        "penguin_data": penguin_data
    }
    
    return result

if __name__ == "__main__":
    # Test with sample data
    test_penguin = {
        "bill_length_mm": 38.7,
        "bill_depth_mm": 18.3,
        "flipper_length_mm": 185.0,
        "body_mass_g": 3800.0
    }
    
    match_result = match_character(test_penguin)
    
    print(f"Best match: {match_result['best_match']['name']} "
          f"with {match_result['best_match']['probability']:.1f}% probability")
    
    print("\nAll matches:")
    for char in match_result['all_matches']:
        print(f"- {char['name']}: {char['probability']:.1f}%")
