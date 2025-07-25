#!/usr/bin/env python3
"""
AgriLens Application Launcher
Comprehensive smart farming assistant with AI-powered features
"""

import os
import sys
import subprocess
import platform

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import streamlit
        import tensorflow
        import numpy
        import pandas
        import requests
        import plotly
        import PIL
        import joblib
        import groq
        from fpdf import FPDF
        
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_models():
    """Check if all required model files exist"""
    required_files = [
        "models/apple_model.h5",
        "models/corn_model.h5", 
        "models/crop_classifier_apple_corn_unknown.h5",
        "models/crop_recommendation_model.pkl",
        "data/apple_class_names.npy",
        "data/corn_class_names.npy",
        "data/crop_classifier_classes.npy",
        "data/class_names.npy"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        return False
    else:
        print("✅ All model files are present!")
        return True

def check_api_keys():
    """Check if API keys are configured"""
    secrets_file = ".streamlit/secrets.toml"
    if not os.path.exists(secrets_file):
        print("❌ Secrets file not found!")
        return False
    
    try:
        with open(secrets_file, 'r') as f:
            content = f.read()
            if "gsk_" in content and "502d8628d859f86e0af77481841f9b6f" in content:
                print("✅ API keys are configured!")
                return True
            else:
                print("❌ API keys not properly configured!")
                return False
    except Exception as e:
        print(f"❌ Error reading secrets file: {e}")
        return False

def launch_application():
    """Launch the AgriLens Streamlit application"""
    print("\n🚀 Launching AgriLens Application...")
    print("=" * 50)
    
    # Check system requirements
    print("🔍 Checking system requirements...")
    
    if not check_dependencies():
        return False
    
    if not check_models():
        return False
    
    if not check_api_keys():
        return False
    
    print("\n🎉 All checks passed! Starting AgriLens...")
    print("=" * 50)
    
    # Launch Streamlit app
    try:
        if platform.system() == "Windows":
            subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        else:
            subprocess.run(["python", "-m", "streamlit", "run", "streamlit_app.py"])
        return True
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        return False

if __name__ == "__main__":
    print("🌾 Welcome to AgriLens - Smart Farming Assistant")
    print("=" * 50)
    print("Features:")
    print("🔍 Disease Detection - AI-powered crop disease identification")
    print("🌱 Crop Recommendations - ML-based planting suggestions")
    print("🌦️ Weather Dashboard - Real-time weather monitoring")
    print("🤖 AI Chatbot - Intelligent farming assistant")
    print("=" * 50)
    
    success = launch_application()
    
    if success:
        print("\n✅ AgriLens launched successfully!")
        print("🌐 Open your browser and navigate to the URL shown above")
        print("🎯 Enjoy your smart farming experience!")
    else:
        print("\n❌ Failed to launch AgriLens")
        print("🔧 Please resolve the issues above and try again")
