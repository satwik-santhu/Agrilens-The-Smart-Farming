# 1. Streamlit config (MUST BE FIRST)
import streamlit as st
st.set_page_config(
    page_title="AgriLens",
    layout="wide",
    page_icon="🌱",
    initial_sidebar_state="expanded"
)

# 2. Environment config
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Disable TensorFlow warnings and info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# 3. Other imports
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from fpdf import FPDF
from PIL import Image
import base64
import tempfile
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re # Import regex module for string manipulation
import json
import random
import io # Import io module for in-memory file handling

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    st.warning("Groq not installed. Using basic chatbot. Install with: pip install groq")

# Import translation library
try:
    from googletrans import Translator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    st.warning("Translation library not installed. Using English only. Install with: pip install googletrans-py")

# Initialize translator with caching
@st.cache_resource
def get_translator():
    """Initialize and cache translator instance"""
    return Translator() if TRANSLATOR_AVAILABLE else None

translator = get_translator()

# Supported languages
SUPPORTED_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn"
}

# Language-specific translations for common UI elements
UI_TRANSLATIONS = {
    "en": {
        "language_selector": "Select Language",
        "home": "🏠 Home",
        "disease_detection": "🔍 Disease Detection",
        "crop_recommendation": "🌱 Crop Recommendation",
        "weather_dashboard": "🌦️ Weather Dashboard",
        "ai_chatbot": "💬 AI Chatbot",
        "welcome": "🌿 Welcome to AgriLens",
        "smart_assistant": "Your Smart Crop Assistant",
        "empowering_farmers": "Empowering farmers with AI-driven agricultural insights",
        "analyze": "Analyze",
        "get_recommendation": "Get Recommendation",
        "quick_questions": "Quick Questions",
        "crop_diseases": "🌱 Crop Diseases",
        "weather_tips": "🌦️ Weather Tips",
        "fertilizers": "🌾 Fertilizers",
        "irrigation": "💧 Irrigation",
        "ask_your_question": "Ask Your Question",
        "send": "Send",
        "clear_chat": "Clear Chat",
        "type_question": "Type your farming question here...",
        "question_placeholder": "e.g., How do I treat apple scab disease?",
        "welcome_message": "Hello! I'm your AgriLens AI assistant. I can help you with farming questions, crop diseases, weather advice, and more. What would you like to know?",
        "helpful_tips": "💡 What I Can Help You With",
        "chatbot_intro": "Ask me anything about farming, crops, diseases, weather, or agricultural practices!",
        "crop_management": "🌱 Crop Management:",
        "weather_climate": "🌦️ Weather & Climate:",
        "specific_crops": "🌾 Specific Crops:",
        "technical_support": "🔬 Technical Support:",
        "pro_tip": "💡 Pro Tip: For detailed disease analysis, use our Disease Detection feature. For weather forecasts, check our Weather Dashboard!"
    },
    "hi": {
        "language_selector": "भाषा चुनें",
        "home": "🏠 होम",
        "disease_detection": "🔍 रोग पहचान",
        "crop_recommendation": "🌱 फसल अनुशंसा",
        "weather_dashboard": "🌦️ मौसम डैशबोर्ड",
        "ai_chatbot": "🤖 एआई चैटबॉट",
        "welcome": "🌿 एग्रीलेंस में आपका स्वागत है",
        "smart_assistant": "आपका स्मार्ट फसल सहायक",
        "empowering_farmers": "किसानों को एआई-संचालित कृषि अंतर्दृष्टि के साथ सशक्त बनाना",
        "analyze": "विश्लेषण करें",
        "get_recommendation": "अनुशंसा प्राप्त करें",
        "quick_questions": "त्वरित प्रश्न",
        "crop_diseases": "🌱 फसल रोग",
        "weather_tips": "🌦️ मौसम टिप्स",
        "fertilizers": "🌾 उर्वरक",
        "irrigation": "💧 सिंचाई",
        "ask_your_question": "अपना प्रश्न पूछें",
        "send": "भेजें",
        "clear_chat": "चैट साफ़ करें",
        "type_question": "यहां अपना कृषि प्रश्न लिखें...",
        "question_placeholder": "उदाहरण, मैं सेब के स्कैब रोग का इलाज कैसे करूं?",
        "welcome_message": "नमस्ते! मैं आपका एग्रीलेंस एआई सहायक हूं। मैं आपको कृषि प्रश्नों, फसल रोगों, मौसम सलाह और अधिक में मदद कर सकता हूं। आप क्या जानना चाहेंगे?",
        "helpful_tips": "💡 मैं आपकी किस प्रकार सहायता कर सकता हूं",
        "chatbot_intro": "मुझसे खेती, फसलों, बीमारियों, मौसम या कृषि प्रथाओं के बारे में कुछ भी पूछें!",
        "crop_management": "🌱 फसल प्रबंधन:",
        "weather_climate": "🌦️ मौसम और जलवायु:",
        "specific_crops": "🌾 विशिष्ट फसलें:",
        "technical_support": "🔬 तकनीकी सहायता:",
        "pro_tip": "💡 प्रो टिप: विस्तृत रोग विश्लेषण के लिए, हमारी रोग पहचान सुविधा का उपयोग करें। मौसम पूर्वानुमान के लिए, हमारा मौसम डैशबोर्ड देखें!"
    },
    "kn": {
        "language_selector": "ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        "home": "🏠 ಮುಖಪುಟ",
        "disease_detection": "🔍 ರೋಗ ಪತ್ತೆ",
        "crop_recommendation": "🌱 ಬೆಳೆ ಶಿಫಾರಸು",
        "weather_dashboard": "🌦️ ಹವಾಮಾನ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "ai_chatbot": "🤖 ಎಐ ಚಾಟ್‌ಬಾಟ್",
        "welcome": "🌿 ಅಗ್ರಿಲೆನ್ಸ್‌ಗೆ ಸುಸ್ವಾಗತ",
        "smart_assistant": "ನಿಮ್ಮ ಸ್ಮಾರ್ಟ್ ಬೆಳೆ ಸಹಾಯಕ",
        "empowering_farmers": "ಎಐ-ಆಧಾರಿತ ಕೃಷಿ ಒಳನೋಟಗಳೊಂದಿಗೆ ರೈತರನ್ನು ಸಬಲೀಕರಣಗೊಳಿಸುವುದು",
        "analyze": "ವಿಶ್ಲೇಷಿಸಿ",
        "quick_questions": "ತ್ವರಿತ ಪ್ರಶ್ನೆಗಳು",
        "crop_diseases": "🌱 ಬೆಳೆ ರೋಗಗಳು",
        "weather_tips": "🌦️ ಹವಾಮಾನ ಸಲಹೆಗಳು",
        "fertilizers": "🌾 ರಸಗೊಬ್ಬರಗಳು",
        "irrigation": "💧 ನೀರಾವರಿ",
        "ask_your_question": "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ",
        "get_recommendation": "ಶಿಫಾರಸು ಪಡೆಯಿರಿ",
        "send": "ಕಳುಹಿಸಿ",
        "clear_chat": "ಚಾಟ್ ಅಳಿಸಿ",
        "type_question": "ಇಲ್ಲಿ ನಿಮ್ಮ ಕೃಷಿ ಪ್ರಶ್ನೆಯನ್ನು ಟೈಪ್ ಮಾಡಿ...",
        "question_placeholder": "ಉದಾ., ನಾನು ಆಪಲ್ ಸ್ಕ್ಯಾಬ್ ರೋಗವನ್ನು ಹೇಗೆ ಚಿಕಿತ್ಸೆ ಮಾಡಬೇಕು?",
        "welcome_message": "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಅಗ್ರಿಲೆನ್ಸ್ ಎಐ ಸಹಾಯಕ. ನಾನು ನಿಮಗೆ ಕೃಷಿ ಪ್ರಶ್ನೆಗಳು, ಬೆಳೆ ರೋಗಗಳು, ಹವಾಮಾನ ಸಲಹೆಗಳು ಮತ್ತು ಇನ್ನೂ ಹೆಚ್ಚಿನದರಲ್ಲಿ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ. ನೀವು ಏನನ್ನು ತಿಳಿಯಲು ಬಯಸುತ್ತೀರಿ?",
        "helpful_tips": "💡 ನಾನು ನಿಮಗೆ ಯಾವ ರೀತಿಯಲ್ಲಿ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ",
        "chatbot_intro": "ಕೃಷಿ, ಬೆಳೆಗಳು, ರೋಗಗಳು, ಹವಾಮಾನ ಅಥವಾ ಕೃಷಿ ಅಭ್ಯಾಸಗಳ ಬಗ್ಗೆ ನನ್ನನ್ನು ಏನಾದರೂ ಕೇಳಿ!",
        "crop_management": "🌱 ಬೆಳೆ ನಿರ್ವಹಣೆ:",
        "weather_climate": "🌦️ ಹವಾಮಾನ ಮತ್ತು ವಾತಾವರಣ:",
        "specific_crops": "🌾 ನಿರ್ದಿಷ್ಟ ಬೆಳೆಗಳು:",
        "technical_support": "🔬 ತಾಂತ್ರಿಕ ಬೆಂಬಲ:",
        "pro_tip": "💡 ಪ್ರೊ ಟಿಪ್: ವಿವರವಾದ ರೋಗ ವಿಶ್ಲೇಷಣೆಗಾಗಿ, ನಮ್ಮ ರೋಗ ಪತ್ತೆ ವೈಶಿಷ್ಟ್ಯವನ್ನು ಬಳಸಿ. ಹವಾಮಾನ ಮುನ್ಸೂಚನೆಗಳಿಗಾಗಿ, ನಮ್ಮ ಹವಾಮಾನ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್ ಪರಿಶೀಲಿಸಿ!"
    }
}

# Function to translate text
@st.cache_data(ttl=3600)  # Cache translations for 1 hour
def translate_text(text, target_language="en"):
    """Translates text to the target language."""
    if not TRANSLATOR_AVAILABLE or target_language == "en":
        return text
    
    try:
        translation = translator.translate(text, dest=target_language)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Function to translate text from non-English to English
def translate_to_english(text, source_language):
    """Translates text from the source language to English."""
    if not TRANSLATOR_AVAILABLE or source_language == "en":
        return text
    
    try:
        translation = translator.translate(text, src=source_language, dest="en")
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Function to get UI text based on selected language
def get_ui_text(key, language_code="en"):
    """Returns the UI text for the given key in the selected language."""
    if language_code in UI_TRANSLATIONS and key in UI_TRANSLATIONS[language_code]:
        return UI_TRANSLATIONS[language_code][key]
    return UI_TRANSLATIONS["en"].get(key, key)

# Disease details mapping
# Keys are standardized to match the output of the new normalize_key_for_details function
# (e.g., "Crop_Disease_Name" with no parentheses or multiple underscores)
DISEASE_DETAILS = {
    "Apple_Scab": {
        "growth_stage": "Early Spring",
        "cause": "Fungal spores in cool, wet weather",
        "nutrient_deficiency": "Iron, Boron, Manganese, Zinc deficiency",
        "solution": "Prune tree; apply fungicides",
        "fertilizer": "Micronutrient mix (Fe, B, Mn, Zn)"
    },
    "Apple_Black_Rot": {
        "growth_stage": "Late Spring",
        "cause": "Fungal infection, warm/humid conditions",
        "nutrient_deficiency": "Potassium, Calcium deficiency",
        "solution": "Remove infected fruit; use fungicides",
        "fertilizer": "Potassium-rich fertilizer"
    },
    "Apple_Cedar_Rust": {
        "growth_stage": "Spring",
        "cause": "Fungal spores from nearby junipers",
        "nutrient_deficiency": "Magnesium, Sulfur deficiency",
        "solution": "Remove nearby junipers; apply fungicides",
        "fertilizer": "Magnesium sulfate (Epsom salt)"
    },
    "Apple_healthy": {
        "growth_stage": "-",
        "cause": "-",
        "nutrient_deficiency": "-",
        "solution": "Maintain regular care",
        "fertilizer": "Balanced NPK fertilizer"
    },
    "Corn_Maize_Common_Rust": { # Standardized key for Corn Common Rust
        "growth_stage": "Mid-Summer",
        "cause": "Fungal spores in warm, humid weather",
        "nutrient_deficiency": "Nitrogen, Phosphorus deficiency",
        "solution": "Apply fungicides; crop rotation",
        "fertilizer": "High-nitrogen fertilizer"
    },
    "Corn_Maize_Gray_Leaf_Spot": { # Standardized key for Corn Gray Leaf Spot
        "growth_stage": "Late Summer",
        "cause": "Fungal spores in warm, humid weather",
        "nutrient_deficiency": "Potassium, Magnesium deficiency",
        "solution": "Remove infected leaves; apply fungicides",
        "fertilizer": "Potassium-rich fertilizer"
    },
    "Corn_Maize_Northern_Leaf_Blight": { # Standardized key for Corn Northern Leaf Blight
        "growth_stage": "Mid-Summer",
        "cause": "Fungal spores in warm, humid weather",
        "nutrient_deficiency": "Zinc, Manganese deficiency",
        "solution": "Crop rotation; apply fungicides",
        "fertilizer": "Micronutrient mix (Zn, Mn)"
    },
    "Corn_Maize_healthy": { # Standardized key for Corn healthy
        "growth_stage": "-",
        "cause": "-",
        "nutrient_deficiency": "-",
        "solution": "Maintain regular care",
        "fertilizer": "Balanced NPK fertilizer"
    },
    "Corn_Maize_Cercospora_Leaf_Spot": {
        "growth_stage": "Mid-Summer",
        "cause": "Fungal spores in warm, humid weather",
        "nutrient_deficiency": "Nitrogen, Potassium deficiency",
        "solution": "Remove infected leaves; apply fungicides",
        "fertilizer": "High-nitrogen fertilizer"
    },
    "Corn_Maize_Cercospora_leaf_spot_Gray_leaf_spot": {
        "growth_stage": "Mid to Late Summer",
        "cause": "Fungal spores (Cercospora and/or Gray Leaf Spot) in warm, humid weather",
        "nutrient_deficiency": "Nitrogen, Potassium, and Magnesium deficiency",
        "solution": "Remove infected leaves, apply fungicides, and practice crop rotation",
        "fertilizer": "Balanced fertilizer with sufficient N, K, and Magnesium"
    }
}

# Weather API - Use environment variable or Streamlit secrets
API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "502d8628d859f86e0af77481841f9b6f")

# Custom CSS for styling
def local_css(file_name):
    """Loads a local CSS file and applies it to the Streamlit app."""
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        # If CSS file doesn't exist, continue without it
        pass

# Load custom CSS (assuming 'style.css' exists in the same directory)
local_css("style.css")

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_current_weather(location):
    """Fetches current weather data for a given location."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        return {
            "location": data['name'],
            "country": data['sys']['country'],
            "temperature": data['main']['temp'],
            "feels_like": data['main']['feels_like'],
            "humidity": data['main']['humidity'],
            "pressure": data['main']['pressure'],
            "visibility": data.get('visibility', 0) / 1000,  # Convert to km
            "wind_speed": data['wind']['speed'],
            "wind_direction": data['wind'].get('deg', 0),
            "weather_main": data['weather'][0]['main'],
            "weather_description": data['weather'][0]['description'],
            "weather_icon": data['weather'][0]['icon'],
            "clouds": data['clouds']['all'],
            "sunrise": datetime.fromtimestamp(data['sys']['sunrise']),
            "sunset": datetime.fromtimestamp(data['sys']['sunset']),
            "timestamp": datetime.now()
        }
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching current weather data: {e}. Please check the location or your internet connection.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching current weather data: {e}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_forecast_weather(location):
    """Fetches 5-day weather forecast data for a given location."""
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        
        forecast_data = []
        for item in data['list']:
            forecast_data.append({
                'datetime': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'weather': item['weather'][0]['description'],
                'weather_icon': item['weather'][0]['icon'],
                'wind_speed': item['wind']['speed'],
                'clouds': item['clouds']['all'],
                'rain': item.get('rain', {}).get('3h', 0) # Get rain volume in last 3 hours
            })
        
        return forecast_data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching forecast data: {e}. Please check the location or your internet connection.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching forecast data: {e}")
        return None

def get_weather_report(location):
    """
    Fetches a simplified weather report for agricultural recommendations.
    This function is kept for compatibility with existing calls.
    """
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={location}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        rains = []
        # Check for rain in the next 24 hours (8 * 3-hour forecasts)
        for i in data["list"][:8]:
            desc = i["weather"][0]["description"]
            if "rain" in desc.lower():
                rains.append(i["dt_txt"])

        forecast = {
            "location": location,
            "next_24h_rain": len(rains) > 0,
            "rain_times": rains,
            "temperature": f"{data['list'][0]['main']['temp']} °C",
            "humidity": f"{data['list'][0]['main']['humidity']}%",
            "weather_icon": data['list'][0]['weather'][0]['icon']
        }

        forecast["advice"] = (
            "🌧️ Rain expected — watch for fungal issues and plan irrigation accordingly!"
            if forecast["next_24h_rain"]
            else "☀️ Dry weather — monitor irrigation needs and conserve water."
        )
        return forecast
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather report: {e}. Please check the location or your internet connection.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching weather report: {e}")
        return None

def create_temperature_chart(forecast_data):
    """Generates a Plotly chart for temperature trends over 5 days."""
    df = pd.DataFrame(forecast_data)
    df['date'] = df['datetime'].dt.strftime('%m/%d %H:%M')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['temperature'],
        mode='lines+markers',
        name='Temperature',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Temperature Forecast (5 Days)",
        xaxis_title="Date & Time",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_humidity_pressure_chart(forecast_data):
    """Generates a Plotly chart for humidity and pressure trends."""
    df = pd.DataFrame(forecast_data)
    df['date'] = df['datetime'].dt.strftime('%m/%d %H:%M')
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Humidity (%)', 'Pressure (hPa)'),
        vertical_spacing=0.4
    )
    fig.update_layout(showlegend=False)

    # Humidity
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['humidity'], 
                  mode='lines+markers', name='Humidity',
                  line=dict(color='#4ecdc4', width=2)),
        row=1, col=1
    )
    
    # Pressure
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['pressure'], 
                  mode='lines+markers', name='Pressure',
                  line=dict(color='#45b7d1', width=2)),
        row=2, col=1
    )
    
    fig.update_layout(height=500, template="plotly_white")
    fig.update_xaxes(title_text="Date & Time", row=2, col=1)
    
    return fig

def create_weather_summary_chart(forecast_data):
    """Generates a Plotly pie chart summarizing weather conditions."""
    df = pd.DataFrame(forecast_data)
    weather_counts = df['weather'].value_counts()
    
    fig = px.pie(
        values=weather_counts.values,
        names=weather_counts.index,
        title="Weather Conditions Distribution (Next 5 Days)"
    )
    
    fig.update_layout(height=400)
    return fig

def display_weather_dashboard():
    """Displays the interactive weather dashboard page."""
    st.header(f"🌦️ {translate_text('Live Weather Dashboard', st.session_state.language_code)}")
    st.markdown(translate_text("Real-time weather monitoring and forecast for agricultural planning", st.session_state.language_code))
    
    # Location input
    col1, col2 = st.columns([3, 1])
    with col1:
        location = st.text_input(
            f"📍 {translate_text('Enter Location (City, Country)', st.session_state.language_code)}",
            value="Bangalore, India",
            help=translate_text("Enter city name and country for accurate weather data", st.session_state.language_code)
        )
    with col2:
        if st.button(f"🔄 {translate_text('Refresh Data', st.session_state.language_code)}", type="primary"):
            st.rerun() # Rerun the app to fetch fresh data
    
    if location:
        # Get current and forecast weather
        current_weather = get_current_weather(location)
        forecast_data = get_forecast_weather(location)
        
        if current_weather and forecast_data:
            # Current Weather Section
            st.subheader(translate_text("Current Weather Conditions", st.session_state.language_code))
            
            # Current weather cards for key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           padding: 20px; border-radius: 15px; color: white; text-align: center;">
                    <h3>{current_weather['temperature']:.1f}°C</h3>
                    <p>{translate_text("Temperature", st.session_state.language_code)}</p>
                    <small>{translate_text("Feels like", st.session_state.language_code)} {current_weather['feels_like']:.1f}°C</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                           padding: 20px; border-radius: 15px; color: white; text-align: center;">
                    <h3>{current_weather['humidity']}%</h3>
                    <p>{translate_text("Humidity", st.session_state.language_code)}</p>
                    <small>{current_weather['pressure']} hPa ({translate_text("Pressure", st.session_state.language_code)})</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                           padding: 20px; border-radius: 15px; color: white; text-align: center;">
                    <h3>{current_weather['wind_speed']:.1f} m/s</h3>
                    <p>{translate_text("Wind Speed", st.session_state.language_code)}</p>
                    <small>{current_weather['visibility']:.1f} km {translate_text("visibility", st.session_state.language_code)}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                           padding: 20px; border-radius: 15px; color: white; text-align: center;">
                    <h3>{translate_text(current_weather['weather_main'], st.session_state.language_code)}</h3>
                    <p>{translate_text(current_weather['weather_description'].title(), st.session_state.language_code)}</p>
                    <img src="http://openweathermap.org/img/wn/{current_weather['weather_icon']}@2x.png" width="50">
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed current conditions in two columns
            st.subheader(translate_text("Detailed Current Conditions", st.session_state.language_code))
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                    <h4>🌡️ {translate_text("Temperature Details", st.session_state.language_code)}</h4>
                    <p><strong>{translate_text("Current", st.session_state.language_code)}:</strong> {current_weather['temperature']:.1f}°C</p>
                    <p><strong>{translate_text("Feels Like", st.session_state.language_code)}:</strong> {current_weather['feels_like']:.1f}°C</p>
                    <p><strong>{translate_text("Humidity", st.session_state.language_code)}:</strong> {current_weather['humidity']}%</p>
                    <p><strong>{translate_text("Pressure", st.session_state.language_code)}:</strong> {current_weather['pressure']} hPa</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                    <h4>💨 {translate_text("Wind & Visibility", st.session_state.language_code)}</h4>
                    <p><strong>{translate_text("Wind Speed", st.session_state.language_code)}:</strong> {current_weather['wind_speed']:.1f} m/s</p>
                    <p><strong>{translate_text("Wind Direction", st.session_state.language_code)}:</strong> {current_weather['wind_direction']}°</p>
                    <p><strong>{translate_text("Visibility", st.session_state.language_code)}:</strong> {current_weather['visibility']:.1f} km</p>
                    <p><strong>{translate_text("Cloud Cover", st.session_state.language_code)}:</strong> {current_weather['clouds']}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Sunrise and Sunset times
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%); 
                       padding: 15px; border-radius: 10px; margin: 20px 0;">
                <h4>🌅 {translate_text("Sun Times", st.session_state.language_code)}</h4>
                <div style="display: flex; justify-content: space-around;">
                    <div><strong>{translate_text("Sunrise", st.session_state.language_code)}:</strong> {current_weather['sunrise'].strftime('%H:%M')}</div>
                    <div><strong>{translate_text("Sunset", st.session_state.language_code)}:</strong> {current_weather['sunset'].strftime('%H:%M')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Forecast Charts Section
            st.subheader(translate_text("Weather Forecast Charts", st.session_state.language_code))
            
            # Temperature chart
            temp_chart = create_temperature_chart(forecast_data)
            st.plotly_chart(temp_chart, use_container_width=True)
            
            # Humidity and Pressure charts side-by-side
            
            
            humidity_pressure_chart = create_humidity_pressure_chart(forecast_data)
            st.plotly_chart(humidity_pressure_chart, use_container_width=True)
            
            weather_summary_chart = create_weather_summary_chart(forecast_data)
            st.plotly_chart(weather_summary_chart, use_container_width=True)
            
            st.markdown("---")
            
            # Agricultural Recommendations based on current weather
            st.subheader(f"🌾 {translate_text('Agricultural Recommendations', st.session_state.language_code)}")
            
            recommendations = []
            
            if current_weather['temperature'] > 30:
                recommendations.append(f"🌡️ **{translate_text('High Temperature Alert', st.session_state.language_code)}**: {translate_text('Consider providing shade for sensitive crops and increase irrigation frequency to prevent heat stress.', st.session_state.language_code)}")
            elif current_weather['temperature'] < 5:
                recommendations.append(f"❄️ **{translate_text('Low Temperature Alert', st.session_state.language_code)}**: {translate_text('Protect crops from frost damage. Consider covering sensitive plants or using irrigation for frost protection.', st.session_state.language_code)}")
            
            if current_weather['humidity'] > 80:
                recommendations.append(f"💧 **{translate_text('High Humidity', st.session_state.language_code)}**: {translate_text('Monitor for fungal diseases. Ensure good air circulation around plants and consider preventative fungicide applications.', st.session_state.language_code)}")
            elif current_weather['humidity'] < 30:
                recommendations.append(f"🏜️ **{translate_text('Low Humidity', st.session_state.language_code)}**: {translate_text('Increase irrigation and consider mulching to retain soil moisture and reduce evaporation.', st.session_state.language_code)}")
            
            if current_weather['wind_speed'] > 10:
                recommendations.append(f"💨 **{translate_text('Strong Winds', st.session_state.language_code)}**: {translate_text('Secure tall plants and greenhouses to prevent structural damage. Check for wind burn on leaves regularly.', st.session_state.language_code)}")
            
            # Check for rain in forecast
            rain_forecast = [item for item in forecast_data if item['rain'] > 0]
            if rain_forecast:
                recommendations.append(f"🌧️ **{translate_text('Rain Expected', st.session_state.language_code)}**: {translate_text('Prepare drainage systems to prevent waterlogging and consider delaying pesticide applications that could be washed away.', st.session_state.language_code)}")
            else:
                recommendations.append(f"☀️ **{translate_text('Dry Conditions', st.session_state.language_code)}**: {translate_text('Plan irrigation schedule carefully and monitor soil moisture levels closely to avoid drought stress.', st.session_state.language_code)}")
            
            if recommendations:
                st.markdown(
                    "<div style='color:#111; font-size:1.1rem;'>" + "<br>".join(recommendations) + "</div>",
                    unsafe_allow_html=True
            )
            else:
                st.markdown(
        f"<div style='color:#111; font-size:1.1rem;'>🌱 <b>{translate_text('Optimal Conditions', st.session_state.language_code)}</b>: {translate_text('Current weather conditions are favorable for most agricultural activities. Continue with regular monitoring.', st.session_state.language_code)}</div>",
        unsafe_allow_html=True
    )
            # Hourly forecast table
            st.subheader(f"📊 {translate_text('Detailed Hourly Forecast', st.session_state.language_code)}")
            
            # Create forecast DataFrame for display (next 24 hours)
            df_display = pd.DataFrame(forecast_data[:24]) 
            time_col = translate_text('Time', st.session_state.language_code)
            temp_col = f"{translate_text('Temp', st.session_state.language_code)} (°C)"
            humidity_col = f"{translate_text('Humidity', st.session_state.language_code)} (%)"
            wind_col = f"{translate_text('Wind', st.session_state.language_code)} (m/s)"
            weather_col = translate_text('Weather', st.session_state.language_code)
            rain_col = f"{translate_text('Rain', st.session_state.language_code)} (mm)"
            
            df_display[time_col] = df_display['datetime'].dt.strftime('%m/%d %H:%M')
            df_display[temp_col] = df_display['temperature'].round(1)
            df_display[humidity_col] = df_display['humidity']
            df_display[wind_col] = df_display['wind_speed'].round(1)
            df_display[weather_col] = df_display['weather'].apply(lambda x: translate_text(x.title(), st.session_state.language_code))
            df_display[rain_col] = df_display['rain'].round(1)
            
            st.dataframe(
                df_display[[time_col, temp_col, humidity_col, wind_col, weather_col, rain_col]],
                use_container_width=True
            )
            
            # Weather alerts section
            st.subheader(f"⚠️ {translate_text('Weather Alerts', st.session_state.language_code)}")
            alerts = []
            
            # Check for extreme conditions in forecast for the next 24 hours
            max_temp = max([item['temperature'] for item in forecast_data[:24]])
            min_temp = min([item['temperature'] for item in forecast_data[:24]])
            max_wind = max([item['wind_speed'] for item in forecast_data[:24]])
            total_rain = sum([item['rain'] for item in forecast_data[:24]])
            
            if max_temp > 35:
                alerts.append(f"🔥 **{translate_text('Heat Warning', st.session_state.language_code)}**: {translate_text('Maximum temperature expected', st.session_state.language_code)}: {max_temp:.1f}°C. {translate_text('Take precautions to protect crops from extreme heat.', st.session_state.language_code)}")
            if min_temp < 0:
                alerts.append(f"🧊 **{translate_text('Frost Warning', st.session_state.language_code)}**: {translate_text('Minimum temperature expected', st.session_state.language_code)}: {min_temp:.1f}°C. {translate_text('Implement frost protection measures immediately.', st.session_state.language_code)}")
            if max_wind > 15:
                alerts.append(f"💨 **{translate_text('Wind Warning', st.session_state.language_code)}**: {translate_text('Maximum wind speed expected', st.session_state.language_code)}: {max_wind:.1f} m/s. {translate_text('Secure vulnerable structures and plants.', st.session_state.language_code)}")
            if total_rain > 20:
                alerts.append(f"🌧️ **{translate_text('Heavy Rain Warning', st.session_state.language_code)}**: {translate_text('Total rainfall expected', st.session_state.language_code)}: {total_rain:.1f} mm. {translate_text('Ensure proper drainage to prevent waterlogging and root rot.', st.session_state.language_code)}")
            if alerts:
                st.markdown(
        "<div style='color:#111; font-size:1.1rem;'>" + "<br>".join(alerts) + "</div>",
        unsafe_allow_html=True
    )
            else:
                st.markdown(
        f"<div style='color:#111; font-size:1.1rem;'>✅ {translate_text('No significant weather alerts for the next 24 hours.', st.session_state.language_code)}</div>",
        unsafe_allow_html=True
    )       

@st.cache_resource
def load_model_and_classes(crop):
    """
    Loads the Keras model and class names for a specific crop disease model.
    Cached to avoid reloading on every prediction.
    """
    model_path = f"models/{crop.lower()}_model.h5"
    class_path = f"data/{crop.lower()}_class_names.npy"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Disease model not found for {crop}: {model_path}")
    if not os.path.exists(class_path):
        raise FileNotFoundError(f"Disease class names not found for {crop}: {class_path}")

    # Show loading message for first load only
    with st.spinner(f"Loading {crop} disease detection model..."):
        model = load_model(model_path)
        class_names = np.load(class_path, allow_pickle=True)
    return model, class_names

@st.cache_resource
def load_crop_classifier():
    """
    Loads the general crop classifier model and its class names.
    Returns None if the model is not found. Cached for performance.
    """
    model_path = "models/crop_classifier_apple_corn_unknown.h5"
    class_path = "data/crop_classifier_classes.npy"

    if not os.path.exists(model_path) or not os.path.exists(class_path):
        return None, None # Return None if model files don't exist
    
    with st.spinner("Loading crop classifier model..."):
        model = load_model(model_path)
        class_names = np.load(class_path, allow_pickle=True)
    return model, class_names

def predict_disease(image_path, model, class_names, selected_crop):
    """
    Performs prediction on an uploaded image using a specific disease model.
    """
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0 # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = round(np.max(predictions[0]) * 100, 2)

    # Define a robust mapping for selected crop to its expected class name prefix
    crop_name_mapping = {
        "apple": "Apple",
        "corn": "Corn_(maize)",
        "grape": "Grape",
        "potato": "Potato",
        "tomato": "Tomato"
    }
    
    expected_prefix = crop_name_mapping.get(selected_crop.lower())

    if not expected_prefix or not predicted_class.startswith(expected_prefix):
        return "Incompatible Image", 0.0

    INTERNAL_CONFIDENCE_THRESHOLD = 60.0

    if confidence < INTERNAL_CONFIDENCE_THRESHOLD:
        return "Incompatible Image", 0.0
    
    return predicted_class, confidence

def predict_crop(image_path, model, class_names):
    """
    Performs prediction on an uploaded image using the general crop classifier model.
    """
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = round(np.max(predictions[0]) * 100, 2)
    
    # Assuming the classes are 'Apple', 'Corn', 'Unknown'
    return predicted_class, confidence

class PDF(FPDF):
    """Custom PDF class for generating reports."""
    def header(self):
        self.set_font("Arial", "B", 16)
        self.set_text_color(34, 139, 34)  # Forest green color
        self.cell(0, 10, "AgriLens Disease Detection Report", ln=True, align="C")
        if os.path.exists("assets/logo.png"):
            self.image("assets/logo.png", 10, 8, 25)
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def safe_text(text):
    """Encodes text to latin-1 to prevent PDF generation errors with special characters."""
    return text.encode("latin-1", "replace").decode("latin-1")

def generate_pdf(report_data, image_path, out_path="report.pdf"):
    """Generates a PDF report with analysis results and weather information."""
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Report Info Section
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Analysis Results", ln=True)
    pdf.set_font("Arial", size=12)
    
    pdf.cell(40, 10, txt="Date:", ln=0)
    pdf.cell(0, 10, txt=safe_text(report_data["date"]), ln=True)
    
    pdf.cell(40, 10, txt="Crop:", ln=0)
    pdf.cell(0, 10, txt=safe_text(report_data["crop"]), ln=True)
    
    pdf.cell(40, 10, txt="Plant Status:", ln=0)
    pdf.cell(0, 10, txt=safe_text(report_data["status"]), ln=True)
    
    pdf.cell(40, 10, txt="Disease:", ln=0)
    pdf.cell(0, 10, txt=safe_text(report_data["disease"]), ln=True)
    
    # ADDED Confidence back to PDF
    pdf.cell(40, 10, txt="Confidence:", ln=0)
    pdf.cell(0, 10, txt=safe_text(report_data["confidence"]), ln=True)
    
    pdf.ln(10)

    # Add uploaded image to PDF
    if os.path.exists(image_path):
        pdf.image(image_path, w=80, h=60)
        pdf.ln(10)

    # Weather section in PDF
    if "weather" in report_data:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="Weather Report", ln=True)
        pdf.set_font("Arial", size=12)
        weather = report_data["weather"]
        
        pdf.cell(40, 10, txt="Location:", ln=0)
        pdf.cell(0, 10, txt=safe_text(weather["location"]), ln=True)
        
        pdf.cell(40, 10, txt="Temperature:", ln=0)
        pdf.cell(0, 10, txt=safe_text(weather["temperature"]), ln=True)
        
        pdf.cell(40, 10, txt="Humidity:", ln=0)
        pdf.cell(0, 10, txt=safe_text(weather["humidity"]), ln=True)
        
        pdf.cell(40, 10, txt="Rain Expected:", ln=0)
        pdf.cell(0, 10, txt="Yes" if weather["next_24h_rain"] else "No", ln=True)
        
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, txt="Recommendation:", ln=0)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=safe_text(weather["advice"]))

    pdf.output(out_path)
    return out_path

def get_weather_icon(icon_code):
    """Returns the URL for a weather icon from OpenWeatherMap."""
    return f"http://openweathermap.org/img/wn/{icon_code}@2x.png"

def main():
    """Main function to run the Streamlit application."""
    # Initialize session state for language if not already set
    if 'language' not in st.session_state:
        st.session_state.language = "English"
        st.session_state.language_code = "en"
    
    # Sidebar with logo, language selector, and navigation
    with st.sidebar:
        if os.path.exists("assets/logo.png"):
            st.image("assets/logo.png", width=150)
        st.title("AgriLens")
        
        # Language selector
        selected_language = st.selectbox(
            get_ui_text("language_selector", st.session_state.language_code),
            list(SUPPORTED_LANGUAGES.keys()),
            index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state.language)
        )
        
        # Update language if changed
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.session_state.language_code = SUPPORTED_LANGUAGES[selected_language]
            st.rerun()
        
        st.markdown("---")
        
        # Navigation with translated options
        page = st.radio(
            "Navigate",
            [
                get_ui_text("home", st.session_state.language_code),
                get_ui_text("disease_detection", st.session_state.language_code),
                get_ui_text("crop_recommendation", st.session_state.language_code),
                get_ui_text("weather_dashboard", st.session_state.language_code),
                get_ui_text("ai_chatbot", st.session_state.language_code)
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center;">
            <p>{translate_text("Smart farming solutions for modern agriculture", st.session_state.language_code)}</p>
        </div>
        """, unsafe_allow_html=True)

    # Main content area based on selected page
    if page == get_ui_text("home", st.session_state.language_code):
        st.header(get_ui_text("welcome", st.session_state.language_code))
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: #f0f8f0; border-radius: 10px; margin-bottom:10px;">
            <h3 style="color: #2e8b57;">{get_ui_text("smart_assistant", st.session_state.language_code)}</h3>
            <p>{get_ui_text("empowering_farmers", st.session_state.language_code)}</p>
        </div>
        """, unsafe_allow_html=True)

        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px; height: 200px;">
                <h4 style="color: #1e6fbb;">🌱 {translate_text("Crop Health", st.session_state.language_code)}</h4>
                <p>{translate_text("Detect diseases and nutrient deficiencies from leaf images with our advanced AI models.", st.session_state.language_code)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: #fff2e6; padding: 15px; border-radius: 10px; height: 200px;">
                <h4 style="color: #cc7a00;">🌾 {translate_text("Smart Recommendations", st.session_state.language_code)}</h4>
                <p>{translate_text("Get personalized crop suggestions based on your soil conditions and local weather.", st.session_state.language_code)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background-color: #e6ffe6; padding: 15px; border-radius: 10px; height: 200px;">
                <h4 style="color: #2e8b57;">⛅ {translate_text("Weather Integration", st.session_state.language_code)}</h4>
                <p>{translate_text("Receive weather-aware farming advice to optimize your agricultural practices.", st.session_state.language_code)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background-color: #f0e6ff; padding: 15px; border-radius: 10px; height: 200px;">
                <h4 style="color: #2e8b57;">🤖 {translate_text(" AgriLens AI", st.session_state.language_code)}</h4>
                <p>{translate_text("AgriLens AI is a smart farming assistant that helps farmers detect crop diseases, analyze nutrient needs, and recommend the right actions for healthier yields.", st.session_state.language_code)}</p>
            </div>
            """, unsafe_allow_html=True)
    
        st.markdown("---")

        st.subheader(translate_text("How It Works", st.session_state.language_code))
        
        steps = [
            {"icon": "📷", "title": translate_text("Upload Image", st.session_state.language_code), "desc": translate_text("Take a clear photo of your crop leaves", st.session_state.language_code)},
            {"icon": "🔍", "title": translate_text("AI Analysis", st.session_state.language_code), "desc": translate_text("Our system detects diseases and nutrient issues", st.session_state.language_code)},
            {"icon": "📊", "title": translate_text("Get Report", st.session_state.language_code), "desc": translate_text("Receive detailed diagnosis and recommendations", st.session_state.language_code)},
            {"icon": "🌦️", "title": translate_text("Weather Monitoring", st.session_state.language_code), "desc": translate_text("Track weather conditions for optimal farming decisions", st.session_state.language_code)}
        ]
        
        cols = st.columns(4)
        for i, step in enumerate(steps):
            with cols[i]:
                st.markdown(f"""
                <div style="text-align: center; padding: 15px;">
                    <div style="font-size: 30px; margin-bottom: 10px;">{step['icon']}</div>
                    <h4>{step['title']}</h4>
                    <p>{step['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

    elif page == get_ui_text("weather_dashboard", st.session_state.language_code):
        display_weather_dashboard()

    elif page == get_ui_text("disease_detection", st.session_state.language_code):
        st.header(translate_text("🔍 Disease & Nutrition Detection", st.session_state.language_code))
        st.markdown(translate_text("Upload an image of your crop leaves to detect diseases or nutrient deficiencies.", st.session_state.language_code))
        
        # --- NEW: Load crop classifier model ---
        crop_classifier_model, crop_classifier_classes = load_crop_classifier()

        with st.expander("📌 Instructions", expanded=True):
            if crop_classifier_model:
                st.markdown("""
                - Upload a clear photo of the plant leaves (Apple or Corn supported)
                - Enter your location for weather-specific advice
                - Our AI will first identify the crop, then analyze it for diseases.
                """)
            else:
                st.markdown("""
                - Select your crop type from the dropdown
                - Enter your location for weather-specific advice
                - Upload a clear photo of the plant leaves
                - Our AI will analyze and provide recommendations
                """)
        
        col1, col2 = st.columns(2)
        
        # Initialize img and temp_image_path outside the conditional block
        img = None 
        temp_image_path = None

        with col1:
            # --- MODIFIED: Conditional Crop Selection ---
            if not crop_classifier_model:
                st.info("Automatic crop classifier not found. Please select a crop manually.")
                crop = st.selectbox(
                    "Select Crop",
                    ["Apple", "Corn", "Grape", "Potato", "Tomato"],
                    help="Choose the crop type you want to analyze"
                )
            else:
                st.success("✅ Automatic crop classifier is active.")
                crop = None # Crop will be determined by the model

            location = st.text_input(
                "📍 Enter Your Location (City, Country)",
                help="This helps us provide weather-specific recommendations"
            )
            
            image_file = st.file_uploader(
                "📤 Upload Leaf Image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of the plant leaves"
            )
        
        with col2:
            if image_file:
                try:
                    # Read the image content into a BytesIO object
                    image_bytes = image_file.getvalue()
                    img = Image.open(io.BytesIO(image_bytes))
                    
                    # Display the image using the PIL Image object
                    # Removed use_container_width=True to resolve TypeError
                    st.image(img, caption=translate_text("Uploaded Leaf Image", st.session_state.language_code))
                except Exception as e:
                    st.error(f"Error loading image for display: {e}")
                    img = None # Ensure img is None if loading fails
        
        if st.button(get_ui_text("analyze", st.session_state.language_code), type="primary", use_container_width=True):
            # Ensure both image and location are provided AND img is a valid PIL Image object
            if img is not None and location: 
                with st.spinner(translate_text("Analyzing your crop...", st.session_state.language_code)):
                    try:
                        # Save the PIL Image object to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            # Convert to RGB if necessary before saving
                            if img.mode in ('RGBA', 'LA'):
                                img = img.convert('RGB')
                            img.save(tmp_file.name, 'JPEG')
                            temp_image_path = tmp_file.name
                        
                        # --- NEW: Two-Step Analysis ---
                        # Step 1: Classify the crop if the model is available
                        if crop_classifier_model:
                            st.write("Step 1: Identifying crop type...")
                            predicted_crop, crop_confidence = predict_crop(temp_image_path, crop_classifier_model, crop_classifier_classes)
                            st.write(f"-> Detected Crop: **{predicted_crop}** (Confidence: {crop_confidence}%)")

                            if predicted_crop.lower() == 'unknown':
                                st.error("❌ The uploaded image could not be identified as a supported crop (Apple or Corn). Please upload a different image.")
                                # Clean up temp file before returning
                                if temp_image_path and os.path.exists(temp_image_path):
                                    try:
                                        os.unlink(temp_image_path)
                                    except Exception as e:
                                        st.warning(f"Could not delete temporary file: {e}")
                                return # Stop analysis
                            
                            # Set the crop for the next step
                            crop = predicted_crop

                        # If crop is still None (manual selection was active but nothing selected)
                        if not crop:
                            st.warning("Please select a crop to analyze.")
                            # Clean up temp file before returning
                            if temp_image_path and os.path.exists(temp_image_path):
                                try:
                                    os.unlink(temp_image_path)
                                except Exception as e:
                                    st.warning(f"Could not delete temporary file: {e}")
                            return

                        # Step 2: Run disease detection on the identified crop
                        st.write(f"Step 2: Analyzing for **{crop}** diseases...")
                        model, class_names = load_model_and_classes(crop)
                        pred_class, confidence = predict_disease(temp_image_path, model, class_names, crop) 
                        
                        # Clean up temp file after prediction
                        if temp_image_path and os.path.exists(temp_image_path):
                            try:
                                os.unlink(temp_image_path)
                            except Exception as e:
                                st.warning(f"Could not delete temporary file: {e}")
                        
                        def normalize_key_for_details(key):
                            key = key.lower()
                            key = key.replace("(", "").replace(")", "")
                            key = key.replace("___", "_")
                            key = key.replace(" ", "_")
                            key = re.sub(r'_+', '_', key)
                            key = key.strip('_')
                            return key

                        lookup_key = normalize_key_for_details(pred_class)
                        mapping_keys = {normalize_key_for_details(k): k for k in DISEASE_DETAILS.keys()}

                        matched_key = mapping_keys.get(lookup_key)
                        if matched_key:
                            details = DISEASE_DETAILS[matched_key]
                        else:
                            details = {
                                "growth_stage": "Not available",
                                "cause": "Not available",
                                "nutrient_deficiency": "Not available",
                                "solution": "Consult a local agricultural expert.",
                                "fertilizer": "General balanced fertilizer"
                            }
                        
                        forecast = get_weather_report(location)
                        
                        if forecast:
                            status = "Healthy" if "healthy" in pred_class.lower() else "Diseased"

                            # ADDED back Confidence-based messages
                            if status == "Diseased":
                                st.markdown(f"""
                                <div style="background-color: #ffe0b2; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
                                    <strong>🚨 Disease Alert!</strong> The model is {confidence}% confident this is <strong>{pred_class}</strong>.
                                </div>
                                """, unsafe_allow_html=True)
                            elif status == "Healthy":
                                st.markdown(f"""
                                <div style="background-color: #c8e6c9; padding: 10px; border-radius: 8px; margin-bottom: 15px;">
                                    <strong>🌿 Plant Status:</strong> {status} with {confidence}% confidence. Your plant appears healthy!
                                </div>
                                """, unsafe_allow_html=True)

                            analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            date, time = analysis_time.split();

                            st.success("Analysis Complete!")
                            
                            result_col1, result_col2 = st.columns(2)
                            
                            with result_col1:
                                st.markdown(f"""
                                <div style="background-color: #f0f8f0; padding: 15px; border-radius: 10px;">
                                    <h3 style="color: #2e8b57;">Results</h3>
                                    <p><strong>Crop:</strong> {crop}</p>
                                    <p><strong>Status:</strong> {status}</p>
                                    <p><strong>Diagnosis:</strong> {pred_class}</p>
                                    <p><strong>Confidence:</strong> {confidence}%</p>
                                    <p><strong>Growth Stage:</strong> {details['growth_stage']}</p>
                                    <p><strong>Cause:</strong> {details['cause']}</p>
                                    <p><strong>Nutrient Deficiency:</strong> {details['nutrient_deficiency']}</p>
                                    <p><strong>Solution:</strong> {details['solution']}</p>
                                    <p><strong>Recommended Fertilizer:</strong> {details['fertilizer']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with result_col2:
                                st.markdown(f"""
                                <div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px;">
                                    <h3 style="color: #1e6fbb;">Weather Report</h3>
                                    <div style="display: flex; align-items: center;">
                                        <img src="{get_weather_icon(forecast['weather_icon'])}" width="50">
                                        <div style="margin-left: 10px;">
                                            <p><strong>Location:</strong> {forecast['location']}</p>
                                            <p><strong>Temperature:</strong> {forecast['temperature']}</p>
                                            <p><strong>Humidity:</strong> {forecast['humidity']}</p>
                                        </div>
                                    </div>
                                    <p><strong>Advice:</strong> {forecast['advice']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            report_data = {
                                "date": date,
                                "time": time,
                                "crop": crop,
                                "analysis_type": "Disease Detection",
                                "status": status,
                                "disease": pred_class,
                                "confidence": f"{confidence}%", # ADDED back
                                "weather": forecast,
                            }

                            pdf_path = generate_pdf(report_data, temp_image_path)
                            
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    "📄 Download Full Report (PDF)",
                                    f,
                                    file_name=f"agrilens_report_{crop.lower()}_{date}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                        else:
                            st.warning("Weather data could not be fetched for the provided location. Recommendation is based on soil and climate parameters only.")
                        
                    except FileNotFoundError as e:
                        st.error(f"Required model or class files are missing. Please ensure they are in the 'models/' and 'data/' directories. Error: {str(e)}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during analysis: {str(e)}")
                        st.error(f"Error type: {type(e).__name__}")
                        if hasattr(e, 'args') and e.args:
                            st.error(f"Error message: {e.args[0]}")
            else:
                st.warning("Please upload an image and enter your location to analyze.")

    elif page == get_ui_text("crop_recommendation", st.session_state.language_code):
        st.header(translate_text("🌱 Smart Crop Recommendation", st.session_state.language_code))
        st.markdown(translate_text("Get personalized crop suggestions based on your soil conditions and climate.", st.session_state.language_code))
        
        with st.expander(translate_text("ℹ️ About This Tool", st.session_state.language_code), expanded=True):
            st.markdown(translate_text("""
            Our AI model recommends the best crops to plant based on:
            - Soil nutrient levels (N, P, K)
            - Temperature and humidity
            - Soil pH and rainfall
            - Your local weather conditions
            """, st.session_state.language_code))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Soil Parameters")
            N = st.slider("Nitrogen (N) level", 0, 150, 50, help="Nitrogen content in soil (kg/ha)")
            P = st.slider("Phosphorus (P) level", 0, 150, 50, help="Phosphorus content in soil (kg/ha)")
            K = st.slider("Potassium (K) level", 0, 150, 50, help="Potassium content in soil (kg/ha)")
            ph = st.slider("Soil pH", 0.0, 14.0, 7.0, 0.1, help="Soil pH level (0-14 scale, 7 is neutral)")
        
        with col2:
            st.subheader("Climate Parameters")
            temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0, 0.1, help="Average temperature (°C)")
            humidity = st.slider("Humidity (%)", 0, 100, 60, help="Relative humidity level (%)")
            rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 100.0, 1.0, help="Annual rainfall (mm)")
            location = st.text_input("📍 Your Location (optional)", help="For weather-specific recommendations")
        
        if st.button(get_ui_text("get_recommendation", st.session_state.language_code), type="primary", use_container_width=True):
            with st.spinner(translate_text("Analyzing your soil and climate...", st.session_state.language_code)):
                try:
                    @st.cache_resource
                    def load_crop_recommendation_model():
                        """Load and cache crop recommendation model"""
                        if not os.path.exists("models/crop_recommendation_model.pkl"):
                            return None
                        return joblib.load("models/crop_recommendation_model.pkl")
                    
                    model = load_crop_recommendation_model()
                    if model is None:
                        st.error("Crop recommendation model not found. Please ensure the model file exists in the 'models' directory.")
                        return
                    prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
                    recommended_crop = prediction[0].title()
                    
                    st.success(f"🌾 Recommended Crop: **{recommended_crop}**")
                    
                    if location:
                        forecast = get_weather_report(location)
                        if forecast:
                            st.info(f"**Weather in {location}:** Current temperature: {forecast['temperature']}, Humidity: {forecast['humidity']}")
                            st.info(f"**Weather-based Recommendation:** {forecast['advice']}")
                        else:
                            st.warning("Could not fetch weather data for the provided location. Recommendation is based on soil and climate parameters only.")
                    
                    st.markdown("---")
                    st.subheader(f"About Growing {recommended_crop}")
                    st.info(f"Detailed growing tips for {recommended_crop} will be available here. This section can include information on optimal planting times, soil requirements, pest management, and harvesting techniques.")
                    
                except Exception as e:
                    st.error(f"Error generating recommendation: {str(e)}")
                    st.error(f"Error details: {type(e).__name__}")
                    if hasattr(e, 'args') and e.args:
                        st.error(f"Error message: {e.args[0]}")

    elif page == get_ui_text("ai_chatbot", st.session_state.language_code):
        display_chatbot()

# Chatbot Knowledge Base
CHATBOT_KNOWLEDGE = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings"],
        "responses": [
            "Hello! I'm your AgriLens AI assistant. How can I help you with your farming needs today?",
            "Hi there! I'm here to help with all your agricultural questions. What would you like to know?",
            "Greetings! I'm your smart farming companion. Ask me anything about crops, diseases, or farming practices!"
        ]
    },
    "crop_diseases": {
        "patterns": ["disease", "sick", "pest", "infection", "fungal", "bacterial", "virus", "spots", "blight", "rot", "rust"],
        "responses": [
            "I can help identify crop diseases! Upload an image in our Disease Detection section, or tell me about the symptoms you're seeing.",
            "Disease management is crucial for healthy crops. What specific symptoms are you observing on your plants?",
            "Common crop diseases include fungal infections, bacterial spots, and viral diseases. Can you describe what you're seeing?"
        ]
    },
    "weather": {
        "patterns": ["weather", "rain", "temperature", "humidity", "forecast", "climate", "drought", "frost"],
        "responses": [
            "Weather plays a vital role in farming! Check our Weather Dashboard for real-time conditions and forecasts.",
            "I can help you understand how weather affects your crops. What weather concerns do you have?",
            "Weather monitoring is essential for successful farming. Are you concerned about specific weather conditions?"
        ]
    },
    "fertilizer": {
        "patterns": ["fertilizer", "nutrition", "nutrients", "NPK", "nitrogen", "phosphorus", "potassium", "deficiency"],
        "responses": [
            "Proper nutrition is key to healthy crops! For corn, use high-nitrogen fertilizers. For fruits, balanced NPK works well.",
            "Nutrient deficiencies show specific symptoms. Yellowing leaves often indicate nitrogen deficiency, while purple leaves suggest phosphorus deficiency.",
            "Fertilizer recommendations depend on your crop and soil conditions. What crop are you growing?"
        ]
    },
    "irrigation": {
        "patterns": ["water", "irrigation", "watering", "drought", "moisture", "dry", "wet"],
        "responses": [
            "Proper irrigation is crucial! Most crops need 1-2 inches of water per week. Check soil moisture before watering.",
            "Overwatering can be as harmful as underwatering. Ensure good drainage to prevent root rot.",
            "Drip irrigation is most efficient for water conservation. What's your current irrigation method?"
        ]
    },
    "planting": {
        "patterns": ["plant", "planting", "sowing", "seed", "germination", "spacing", "depth"],
        "responses": [
            "Planting timing is crucial! Most crops have specific seasons. What crop are you planning to plant?",
            "Seed depth should be 2-3 times the seed diameter. Proper spacing ensures good air circulation.",
            "Soil temperature and moisture are key for germination. Test your soil before planting."
        ]
    },
    "harvest": {
        "patterns": ["harvest", "harvesting", "picking", "mature", "ripe", "ready"],
        "responses": [
            "Harvest timing affects quality and yield. Look for visual cues like color change and firmness.",
            "Different crops have different harvest indicators. What crop are you ready to harvest?",
            "Post-harvest handling is crucial for quality. Ensure proper storage and handling techniques."
        ]
    },
    "soil": {
        "patterns": ["soil", "pH", "acidity", "alkaline", "compost", "organic matter", "soil test"],
        "responses": [
            "Soil health is fundamental! Most crops prefer pH 6.0-7.0. Have you tested your soil recently?",
            "Organic matter improves soil structure and nutrient retention. Consider adding compost regularly.",
            "Soil testing reveals nutrient levels and pH. This helps in making informed fertilizer decisions."
        ]
    },
    "organic_farming": {
        "patterns": ["organic", "natural", "pesticide-free", "chemical-free", "sustainable"],
        "responses": [
            "Organic farming uses natural methods! Crop rotation, beneficial insects, and organic fertilizers are key.",
            "Sustainable practices protect the environment and produce healthier food. What organic methods interest you?",
            "Natural pest control includes companion planting, beneficial insects, and organic sprays."
        ]
    },
    "apple_specific": {
        "patterns": ["apple", "apple tree", "orchard", "apple disease", "apple pest"],
        "responses": [
            "Apple trees are susceptible to scab, fire blight, and cedar rust. Regular pruning and fungicide applications help.",
            "Apple orchards need cross-pollination for fruit set. Plant different varieties nearby.",
            "Common apple pests include codling moth and aphids. Integrated pest management works best."
        ]
    },
    "corn_specific": {
        "patterns": ["corn", "maize", "corn field", "corn disease", "corn pest"],
        "responses": [
            "Corn is prone to northern leaf blight and gray leaf spot. Crop rotation helps prevent disease buildup.",
            "Corn needs high nitrogen levels. Apply nitrogen fertilizer at planting and side-dress during growth.",
            "Common corn pests include corn borer and rootworm. Monitor regularly and use appropriate treatments."
        ]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you", "thanks", "thank you", "exit", "quit"],
        "responses": [
            "Goodbye! Feel free to come back anytime for more farming advice. Happy farming!",
            "Thank you for using AgriLens! Wishing you a successful harvest!",
            "See you later! Remember, I'm always here to help with your agricultural needs."
        ]
    }
}

@st.cache_data(ttl=1800)  # Cache AI responses for 30 minutes
def get_ai_response(user_input, language_code="en"):
    """Generate AI-powered chatbot response using Groq API."""
    if not GROQ_AVAILABLE:
        return get_fallback_response(user_input, language_code)
    
    try:
        # If language is not English, translate input to English for the API
        original_language = language_code
        english_input = user_input
        if language_code != "en":
            english_input = translate_to_english(user_input, language_code)
        
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key or groq_api_key == "your_groq_api_key_here":
            return get_fallback_response(english_input, language_code)
        
        client = Groq(api_key=groq_api_key)
        
        # Create agricultural-focused system prompt
        system_prompt = """
You are AgriLens AI, a specialized agricultural assistant. You help farmers with:
- Crop diseases and pest management
- Weather-related farming advice
- Fertilizer and soil recommendations
- Irrigation and water management
- Planting and harvesting guidance
- Organic farming practices
- Apple and corn cultivation specifics
- The responce should be under 200 words and should not be overwhelming for farmer.

Provide practical, actionable advice. Keep responses concise but informative. 
Always be encouraging and supportive to farmers. If you're unsure about something, 
suggest consulting local agricultural experts or using AgriLens features like Disease Detection or Weather Dashboard.
"""
        
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": english_input}
            ],
            temperature=0.7,
            max_tokens=300
        )
        
        response = completion.choices[0].message.content
        
        # Translate response back to original language if needed
        if original_language != "en":
            response = translate_text(response, original_language)
        
        return response
        
    except Exception as e:
        st.error(f"AI service temporarily unavailable: {str(e)}")
        return get_fallback_response(user_input, language_code)

def get_fallback_response(user_input, language_code="en"):
    """Generate fallback response using pattern matching when AI is unavailable."""
    # If language is not English, translate input to English for pattern matching
    english_input = user_input
    if language_code != "en":
        english_input = translate_to_english(user_input, language_code)
    
    english_input = english_input.lower().strip()
    
    # Check for patterns in user input
    for category, data in CHATBOT_KNOWLEDGE.items():
        for pattern in data["patterns"]:
            if pattern in english_input:
                response = random.choice(data["responses"])
                # Translate response back to original language if needed
                if language_code != "en":
                    response = translate_text(response, language_code)
                return response
    
    # Default responses if no pattern matches
    default_responses = [
        "I'm here to help with farming questions! You can ask me about crop diseases, weather, fertilizers, irrigation, or specific crops like apple and corn.",
        "I specialize in agricultural advice. Try asking about plant diseases, soil conditions, weather impacts, or farming techniques.",
        "Feel free to ask me about crop management, pest control, fertilization, or use our other AgriLens features for detailed analysis!"
    ]
    
    response = random.choice(default_responses)
    
    # Translate response back to original language if needed
    if language_code != "en":
        response = translate_text(response, language_code)
    
    return response

def display_chatbot():
    """Display the AI chatbot interface."""
    st.header(f"🤖 {get_ui_text('ai_chatbot', st.session_state.language_code)}")
    st.markdown(get_ui_text("chatbot_intro", st.session_state.language_code))
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        welcome_message = get_ui_text("welcome_message", st.session_state.language_code)
        st.session_state.chat_history = [
            {"role": "assistant", "content": welcome_message}
        ]
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 10px 0; text-align: right;">
                    <strong>You:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f1f8e9; padding: 10px; border-radius: 10px; margin: 10px 0;">
                    <strong>🤖 AgriLens AI:</strong> {message['content']}
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    st.markdown(f"### {get_ui_text('ask_your_question', st.session_state.language_code)}")
    user_input = st.text_input(
        get_ui_text("type_question", st.session_state.language_code),
        placeholder=get_ui_text("question_placeholder", st.session_state.language_code),
        key="user_input"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button(get_ui_text("send", st.session_state.language_code), type="primary", use_container_width=True):
            if user_input:
                # Add user message
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Generate and add bot response
                response = get_ai_response(user_input, st.session_state.language_code)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Clear input and refresh
                st.rerun()
    
    with col2:
        if st.button(get_ui_text("clear_chat", st.session_state.language_code), use_container_width=True):
            welcome_message = get_ui_text("welcome_message", st.session_state.language_code)
            st.session_state.chat_history = [
                {"role": "assistant", "content": welcome_message}
            ]
            st.rerun()
    
    # Quick action buttons
    st.markdown(f"### {get_ui_text('quick_questions', st.session_state.language_code)}")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button(get_ui_text("crop_diseases", st.session_state.language_code), use_container_width=True):
            user_question = translate_text("Tell me about crop diseases", st.session_state.language_code)
            response = get_ai_response(user_question, st.session_state.language_code)
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col2:
        if st.button(get_ui_text("weather_tips", st.session_state.language_code), use_container_width=True):
            user_question = translate_text("How does weather affect farming?", st.session_state.language_code)
            response = get_ai_response(user_question, st.session_state.language_code)
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col3:
        if st.button(get_ui_text("fertilizers", st.session_state.language_code), use_container_width=True):
            user_question = translate_text("What fertilizers should I use?", st.session_state.language_code)
            response = get_ai_response(user_question, st.session_state.language_code)
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    with col4:
        if st.button(get_ui_text("irrigation", st.session_state.language_code), use_container_width=True):
            user_question = translate_text("How should I water my crops?", st.session_state.language_code)
            response = get_ai_response(user_question, st.session_state.language_code)
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    # Helpful tips section
    st.markdown("---")
    st.markdown(f"### {get_ui_text('helpful_tips', st.session_state.language_code)}")
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown(f"""
        **{get_ui_text('crop_management', st.session_state.language_code)}**
        - Disease identification and treatment
        - Pest control strategies
        - Nutrient deficiency diagnosis
        - Planting and harvesting advice
        
        **{get_ui_text('weather_climate', st.session_state.language_code)}**
        - Weather impact on crops
        - Irrigation scheduling
        - Seasonal farming tips
        - Climate adaptation strategies
        """)
    
    with tips_col2:
        st.markdown(f"""
        **{get_ui_text('specific_crops', st.session_state.language_code)}**
        - Apple orchard management
        - Corn/Maize cultivation
        - Soil preparation techniques
        - Organic farming practices
        
        **{get_ui_text('technical_support', st.session_state.language_code)}**
        - Fertilizer recommendations
        - Soil testing interpretation
        - Sustainable farming methods
        - Crop rotation planning
        """)
    
    st.markdown("---")
    st.info(get_ui_text("pro_tip", st.session_state.language_code))

if __name__ == "__main__":
    main()
