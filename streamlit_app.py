"""
TRU Salary Predictor - Main Deployment App
Copyright (c) 2025 TRU Salary Predictor
Contact: bhuskutetrupti@gmail.com

Main deployment file for the TRU Salary Predictor application.
This file serves as the entry point for deployment platforms.
"""

import streamlit as st
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main app from polished_salary_app
from polished_salary_app import main

if __name__ == "__main__":
    main() 