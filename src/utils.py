"""
utils.py - Helper utilities for the Crop Yield & Price Prediction System.

Contains shared constants, path helpers, and plotting utilities used across
multiple modules in the project.
"""

import os
import sys

# ─── Project Paths ──────────────────────────────────────────────────────────────

# Resolve project root regardless of where the script is invoked from
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
VISUALIZATIONS_DIR = os.path.join(PROJECT_ROOT, "visualizations")


def ensure_dirs():
    """Create all required project directories if they don't exist."""
    for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, VISUALIZATIONS_DIR]:
        os.makedirs(d, exist_ok=True)


# ─── Shared Constants ───────────────────────────────────────────────────────────

# Indian states used in dataset generation
STATES_AND_DISTRICTS = {
    "Andhra Pradesh": ["Guntur", "Krishna", "East Godavari", "West Godavari", "Kurnool"],
    "Telangana": ["Warangal", "Nizamabad", "Karimnagar", "Medak", "Nalgonda"],
    "Tamil Nadu": ["Thanjavur", "Coimbatore", "Madurai", "Salem", "Tirunelveli"],
    "Karnataka": ["Belgaum", "Dharwad", "Shimoga", "Mysore", "Raichur"],
    "Maharashtra": ["Pune", "Nashik", "Nagpur", "Kolhapur", "Ahmednagar"],
    "Uttar Pradesh": ["Lucknow", "Agra", "Varanasi", "Meerut", "Allahabad"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda"],
    "Madhya Pradesh": ["Indore", "Bhopal", "Jabalpur", "Gwalior", "Ujjain"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"],
    "West Bengal": ["Burdwan", "Hooghly", "Murshidabad", "Nadia", "Birbhum"],
    "Gujarat": ["Ahmedabad", "Rajkot", "Surat", "Vadodara", "Junagadh"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga"],
}

CROPS = [
    "Rice", "Wheat", "Maize", "Sugarcane", "Cotton",
    "Soybean", "Groundnut", "Mustard", "Jowar", "Bajra",
    "Tur (Arhar)", "Chana (Gram)", "Onion", "Potato", "Tomato",
]

SEASONS = ["Kharif", "Rabi", "Zaid", "Whole Year"]

SOIL_TYPES = [
    "Alluvial", "Black (Regur)", "Red", "Laterite",
    "Desert (Arid)", "Mountain", "Clay", "Sandy Loam",
]

# Feature columns common to both yield and price datasets
CATEGORICAL_FEATURES = ["State", "District", "Crop", "Season", "Soil_Type"]
NUMERICAL_FEATURES = [
    "Rainfall_mm", "Temperature_C", "Humidity_pct",
    "Area_hectares", "Fertilizer_kg_per_hectare", "Pesticide_kg_per_hectare",
]

YIELD_TARGET = "Yield_ton_per_hectare"
PRICE_TARGET = "Price_per_quintal"


def print_section(title: str):
    """Print a formatted section header to console."""
    line = "=" * 60
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}\n")
