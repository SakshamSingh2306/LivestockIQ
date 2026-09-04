
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import timm
import os
import csv
import pandas as pd
from datetime import datetime
from io import BytesIO
import random
from huggingface_hub import hf_hub_download
import json

# Set page config first
st.set_page_config(page_title="🐄 Cattle Breed Identifier", layout="centered", initial_sidebar_state="collapsed")

# ============ TRANSLATIONS ============
def get_translation(key, language="en"):
    translations = {
        "en": {
            "title": "🐄 Indian Cattle Breed Identifier",
            "subtitle": "Discover the rich diversity of Indian bovine breeds",
            "upload_info": "📁 Upload an image of a cow or buffalo to identify its breed",
            "marketplace_button": "🐂 Cattle Marketplace",
            "upload_label": "Choose a cattle image",
            "analyzing": "🔍 Analyzing breed characteristics...",
            "predicted_breed": "✅ Predicted Breed:",
            "confidence": "🔎 Confidence:",
            "breed_info": "📚 Breed Information",
            "pedigree": "Pedigree / Lineage",
            "productivity": "Productivity",
            "rearing_conditions": "Optimal Rearing Conditions",
            "origin": "Origin",
            "physical_chars": "Physical Characteristics",
            "lifespan": "Lifespan (Years)",
            "temperament": "Temperament",
            "productivity_metrics": "Productivity Metrics",
            "physical_measurements": "📏 Physical Measurements",
            "body_length": "Body Length",
            "height_withers": "Height at Withers",
            "chest_width": "Chest Width",
            "rump_angle": "Rump Angle",
            "refresh": "🔄 Refresh the page to analyze another image",
            "heritage": "🐄 Celebrating India's rich bovine heritage",
            "marketplace_title": "🐂 Cattle Marketplace",
            "marketplace_subtitle": "Buy and Sell Quality Cattle",
            "back_button": "← Back to Breed Identifier",
            "price": "Price",
            "age": "Age",
            "milk_yield": "Milk Yield",
            "lactation_stage": "Lactation Stage",
            "vaccination": "Vaccination",
            "seller": "Seller",
            "contact": "Contact",
            "location": "Location",
            "add_listing": "Add Your Listing",
            "cattle_breed": "Cattle Breed",
            "submit_listing": "Submit Listing",
            "listing_submitted": "Listing submitted!",
            "description": "Description",
            "prediction_error": "❌ Prediction error",
            "processing_error": "⚠ Error processing image",
            "confidence_error": "Could not confidently identify the breed.",
            "no_info": "⚠ No additional information found for this breed.",
            "model_loading": "🔄 Loading AI model...",
            "model_error": "⚠️ Model not found. Using demo mode.",
            "demo_mode": "🔄 Running in demo mode. Upload an image to get a prediction!",
            "chat_title": "Cattle Assistant"
        },
        "hi": {
            "title": "🐄 भारतीय मवेशी नस्ल पहचानकर्ता",
            "subtitle": "भारतीय बोवाइन नस्लों की समृद्ध विविधता की खोज करें",
            "upload_info": "📁 अपनी नस्ल की पहचान करने के लिए गाय या भैंस की एक छवि अपलोड करें",
            "marketplace_button": "🐂 मवेशी बाजार",
            "upload_label": "एक मवेशी छवि चुनें",
            "analyzing": "🔍 नस्ल की विशेषताओं का विश्लेषण किया जा रहा है...",
            "predicted_breed": "✅ अनुमानित नस्ल:",
            "confidence": "🔎 आत्मविश्वास:",
            "breed_info": "📚 नस्ल की जानकारी",
            "pedigree": "वंशावली / वंश",
            "productivity": "उत्पादकता",
            "rearing_conditions": "इष्टतम पालन की स्थिति",
            "origin": "मूल",
            "physical_chars": "शारीरिक विशेषताएं",
            "lifespan": "जीवनकाल (वर्ष)",
            "temperament": "स्वभाव",
            "productivity_metrics": "उत्पादकता मेट्रिक्स",
            "physical_measurements": "📏 शारीरिक माप",
            "body_length": "शरीर की लंबाई",
            "height_withers": "कंधे की ऊंचाई",
            "chest_width": "छाती की चौड़ाई",
            "rump_angle": "रंप कोण",
            "refresh": "🔄 किसी अन्य छवि का विश्लेषण करने के लिए पृष्ठ ताज़ा करें",
            "heritage": "🐄 भारत की समृद्ध बोवाइन विरासत का जश्न",
            "marketplace_title": "🐂 मवेशी बाजार",
            "marketplace_subtitle": "गुणवत्तापूर्ण मवेशी खरीदें और बेचें",
            "back_button": "← ब्रीड आइडेंटिफायर पर वापस जाएं",
            "price": "कीमत",
            "age": "उम्र",
            "milk_yield": "दूध उत्पादन",
            "lactation_stage": "दुग्धावस्था",
            "vaccination": "टीकाकरण",
            "seller": "विक्रेता",
            "contact": "संपर्क",
            "location": "स्थान",
            "add_listing": "अपनी लिस्टिंग जोड़ें",
            "cattle_breed": "मवेशी नस्ल",
            "submit_listing": "लिस्टिंग सबमिट करें",
            "listing_submitted": "लिस्टिंग सबमिट की गई!",
            "description": "विवरण",
            "prediction_error": "❌ भविष्यवाणी त्रुटि",
            "processing_error": "⚠ छवि प्रसंस्करण में त्रुटि",
            "confidence_error": "नस्ल को विश्वास के साथ पहचान नहीं सका।",
            "no_info": "⚠ इस नस्ल के लिए कोई अतिरिक्त जानकारी नहीं मिली।",
            "model_loading": "🔄 AI मॉडल लोड हो रहा है...",
            "model_error": "⚠️ मॉडल नहीं मिला। डेमो मोड में चल रहा है।",
            "demo_mode": "🔄 डेमो मोड में चल रहा है। भविष्यवाणी प्राप्त करने के लिए अपनी छवि अपलोड करें!",
            "chat_title": "मवेशी सहायक"
        },
        "te": {
            "title": "🐄 భారతీయ పశువుల జాతి గుర్తింపు",
            "subtitle": "భారతీయ పశువుల జాతుల సంపన్న వైవిధ్యాన్ని కనుగొనండి",
            "upload_info": "📁 దాని జాతిని గుర్తించడానికి ఒక ఆవు లేదా ఎదురు చిత్రాన్ని అప్లోడ్ చేయండి",
            "marketplace_button": "🐂 పశువుల మార్కెట్",
            "upload_label": "ఒక పశు చిత్రాన్ని ఎంచుకోండి",
            "analyzing": "🔍 జాతి లక్షణాలను విశ్లేషిస్తోంది...",
            "predicted_breed": "✅ అంచనా వేసిన జాతి:",
            "confidence": "🔎 నమ్మకం:",
            "breed_info": "📚 జాతి సమాచారం",
            "pedigree": "వంశం / వంశావళి",
            "productivity": "ఉత్పాదకత",
            "rearing_conditions": "ఆదర్శ పెంపకడ పరిస్థితులు",
            "origin": "మూలం",
            "physical_chars": "భౌతిక లక్షణాలు",
            "lifespan": "ఆయుష్ (సంవత్సరాలు)",
            "temperament": "స్వభావం",
            "productivity_metrics": "ఉత్పాదకత మెట్రిక్స్",
            "physical_measurements": "📏 భౌతిక కొలతలు",
            "body_length": "శరీర పొడవు",
            "height_withers": "భుజాల ఎత్తు",
            "chest_width": "ఛాతీ వెడల్పు",
            "rump_angle": "రంప్ కోణం",
            "refresh": "🔄 మరొక చిత్రాన్ని విశ్లేషించడానికి పేజీని రిఫ్రెష్ చేయండి",
            "heritage": "🐄 భారతదేశం యొక్క సంపన్న పశు వారసత్వాన్ని జరుపుకుంటోంది",
            "marketplace_title": "🐂 పశువుల మార్కెట్",
            "marketplace_subtitle": "నాణ్యత గల పశువులను కొనండి మరియు విక్రయించండి",
            "back_button": "← బ్రీడ్ ఐడెంటిఫైయర్‌కు తిరిగి వెళ్లండి",
            "price": "ధర",
            "age": "వయస్సు",
            "milk_yield": "పాలు దిగుబడి",
            "lactation_stage": "పాల ఉత్పత్తి దశ",
            "vaccination": "తడిపించడం",
            "seller": "విక్రేత",
            "contact": "సంప్రదింపు",
            "location": "స్థానం",
            "add_listing": "మీ లిస్టింగ్‌ని జోడించండి",
            "cattle_breed": "పశు జాతి",
            "submit_listing": "లిస్టింగ్ సమర్పించండి",
            "listing_submitted": "లిస్టింగ్ సమర్పించబడింది!",
            "description": "వివరణ",
            "prediction_error": "❌ అంచనా దోషం",
            "processing_error": "⚠ చిత్ర ప్రాసెసింగ్ లో దోషం",
            "confidence_error": "జాతిని నమ్మకంగా గుర్తించలేకపోయింది.",
            "no_info": "⚠ ఈ జాతి కోసం అదనపు సమాచారం లేదు.",
            "model_loading": "🔄 AI మోడల్ లోడ్ అవుతోంది...",
            "model_error": "⚠️ మోడల్ కనుగొనబడలేదు. డెమో మోడ్‌లో నడుస్తోంది.",
            "demo_mode": "🔄 డెమో మోడ్‌లో నడుస్తోంది. అంచనా పొందడానికి మీ చిత్రాన్ని అప్లోడ్ చేయండి!",
            "chat_title": "పశు సహాయక"
        }
    }
    return translations.get(language, translations["en"]).get(key, key)

def language_selector():
    st.sidebar.markdown("---")
    st.sidebar.header("🌐 Language")
    language = st.sidebar.radio("Select Language", ["English", "Hindi", "Telugu"], index=0, label_visibility="collapsed")
    lang_map = {"English": "en", "Hindi": "hi", "Telugu": "te"}
    return lang_map[language]

# ============ STYLING ============
def set_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background: url('https://images.unsplash.com/photo-1527153857715-3908f2bae5e8?ixlib=rb-4.0.3&auto=format&fit=crop&w=2089&q=80');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.92);
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .main-header {
            color: #2c3e50;
            text-align: center;
            font-size: 2.8rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .sub-header {
            color: #34495e;
            text-align: center;
            font-size: 1.3rem;
            margin-bottom: 2rem;
        }
        .prediction-box {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #3498db;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .breed-info {
            background-color: rgba(248, 249, 250, 0.95);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #27ae60;
            margin: 15px 0;
        }
        .footer {
            text-align: center;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            margin-top: 20px;
        }
        .cattle-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .cattle-name {
            font-weight: bold;
            font-size: 18px;
            color: #2c3e50;
        }
        .cattle-price {
            color: #27ae60;
            font-weight: bold;
            font-size: 16px;
        }
        .seller-info {
            color: #7f8c8d;
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_custom_style()

# ============ LANGUAGE ============
language = language_selector()

# ============ DEVICE ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ BREED LABELS ============
@st.cache_data
def load_breed_labels():

    classes_path = hf_hub_download(
        repo_id="ujjwal75/indian-bovine-breeds-model",
        filename="classes.json"
    )

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    # Handle either a list or dictionary format
    if isinstance(classes, list):
        return classes

    if isinstance(classes, dict):
        # If mapping is index -> class
        try:
            return [
                classes[str(i)]
                for i in range(len(classes))
            ]
        except KeyError:
            # If mapping is class -> index
            return [
                label
                for label, index in sorted(
                    classes.items(),
                    key=lambda x: x[1]
                )
            ]

    raise ValueError("Unsupported classes.json format")

breed_labels = load_breed_labels()
# ============ BREED INFO ============
breed_info_raw = {
    "gir": {
        "info": """ORIGINATED IN GIR FOREST OF GUJARAT
1500-2000 Liters
ADAPTED TO HOT CLIMATES
INDIA (Gujarat)
LARGE SIZE, REDDISH BROWN WITH WHITE SPOTS
12-15
DOCILE AND GENTLE
GOOD MILK YIELD WITH HIGH FAT CONTENT""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "sahiwal": {
        "info": """ORIGINATED IN SAHIWAL DISTRICT, PAKISTAN
2000-3000 Liters
ADAPTED TO TROPICAL CLIMATES
PAKISTAN
MEDIUM SIZE, REDDISH BROWN COLOR
12-15
DOCILE AND HARDY
ONE OF THE BEST DAIRY BREEDS IN TROPICS""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "jersey": {
        "info": """ORIGINATED IN JERSEY ISLAND, UK
5000-6000 Liters
ADAPTED TO VARIOUS CLIMATES
UNITED KINGDOM
SMALL TO MEDIUM SIZE, LIGHT BROWN TO DARK BROWN
10-12
DOCILE AND GENTLE
HIGH EFFICIENCY IN MILK PRODUCTION""",
        "measurements": {
            "body_length": "140-150 cm",
            "height_withers": "130-140 cm",
            "chest_width": "45-50 cm",
            "rump_angle": "6-8 degrees"
        }
    },
    "murrah": {
        "info": """ORIGINATED IN HARYANA, INDIA
1800-2500 Liters
ADAPTED TO NORTH INDIAN CLIMATE
INDIA (Haryana)
MEDIUM SIZE, JET BLACK WITH TIGHT CURLS
12-15
DOCILE AND GENTLE
PREMIUM BUFFALO BREED FOR MILK PRODUCTION""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "holstein_friesian": {
        "info": """ORIGINATED IN NETHERLANDS AND GERMANY
7000-9000 Liters
ADAPTED TO TEMPERATE CLIMATES
NETHERLANDS/GERMANY
LARGE SIZE, BLACK AND WHITE OR RED AND WHITE
10-12
DOCILE AND CALM
HIGHEST MILK PRODUCING DAIRY BREED""",
        "measurements": {
            "body_length": "160-170 cm",
            "height_withers": "150-160 cm",
            "chest_width": "55-60 cm",
            "rump_angle": "4-6 degrees"
        }
    },
    "ongole": {
        "info": """ORIGINATED IN ANDHRA PRADESH, INDIA
NA (Draft breed)
ADAPTED TO TROPICAL CLIMATES
INDIA (Andhra Pradesh)
LARGE SIZE, WHITE TO LIGHT GREY COLOR
15-20
STRONG AND HARDY
PREMIUM DRAFT BREED, EXPORTED WORLDWIDE""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "kankrej": {
        "info": """ORIGINATED IN GUJARAT, INDIA
NA (Draft breed)
ADAPTED TO ARID CLIMATES
INDIA (Gujarat)
LARGE SIZE, GREY TO SILVERY GREY COLOR
15-20
STRONG AND HARDY
ONE OF THE BEST INDIAN DRAFT BREEDS""",
        "measurements": {
            "body_length": "155-165 cm",
            "height_withers": "145-155 cm",
            "chest_width": "52-57 cm",
            "rump_angle": "5-7 degrees"
        }
    },
    "tharparkar": {
        "info": """ORIGINATED IN THARPARKAR DISTRICT, PAKISTAN
1500-2000 Liters
ADAPTED TO DESERT CLIMATES
PAKISTAN
MEDIUM SIZE, WHITE TO LIGHT GREY COLOR
12-15
HARDY AND DOCILE
GOOD MILK YIELD IN ARID CONDITIONS""",
        "measurements": {
            "body_length": "150-160 cm",
            "height_withers": "140-150 cm",
            "chest_width": "50-55 cm",
            "rump_angle": "5-7 degrees"
        }
    }
}

# ============ IMAGE TRANSFORM ============
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============ MODEL LOADING ============
# ============ MODEL LOADING ============

@st.cache_resource
def load_model():

    try:
        with st.spinner(get_translation("model_loading", language)):

            # Download model from Hugging Face
            checkpoint_path = hf_hub_download(
                repo_id="ujjwal75/indian-bovine-breeds-model",
                filename="Indian_bovine_finetuned_model.pth"
            )

            # Create ResNet-50 with 40 output classes
            model = timm.create_model(
                "resnet50",
                pretrained=False,
                num_classes=len(breed_labels)
            )

            # Load trained weights
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False
            )

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

                model.load_state_dict(
                    checkpoint["model_state_dict"]
                )

            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:

                model.load_state_dict(
                    checkpoint["state_dict"]
                )

            else:

                model.load_state_dict(checkpoint)

            model.to(device)
            model.eval()

            return model

    except Exception as e:

        st.error(
            f"Model loading error: {str(e)}"
        )

        return None

model = load_model()

# ============ PREDICTION FUNCTIONS ============
def predict_breed(image):
    try:
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_label = breed_labels[predicted_idx.item()]
            confidence_percent = confidence.item() * 100
        return predicted_label, confidence_percent
    except Exception as e:
        return None, 0

def demo_predict(image):
    breed = random.choice(breed_labels)
    confidence = random.uniform(65, 90)
    return breed, confidence

def save_to_csv(breed, confidence, filename, timestamp):
    try:
        csv_file = "cattle_classification_data.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['timestamp', 'breed', 'confidence', 'filename'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({'timestamp': timestamp, 'breed': breed, 'confidence': confidence, 'filename': filename})
    except:
        pass

# ============ DISPLAY BREED INFO ============
def display_breed_info(breed_key, breed_data, language):
    try:
        lines = breed_data["info"].strip().split("\n")
        if len(lines) < 8:
            return
        
        info_html = f"""
        <div class="breed-info">
            <p>🧬 <b>{get_translation("pedigree", language)}</b>: {lines[0]}</p>
            <p>🍼 <b>{get_translation("productivity", language)}</b>: {lines[1]}</p>
            <p>🌿 <b>{get_translation("rearing_conditions", language)}</b>: {lines[2]}</p>
            <p>🌍 <b>{get_translation("origin", language)}</b>: {lines[3]}</p>
            <p>🐮 <b>{get_translation("physical_chars", language)}</b>: {lines[4]}</p>
            <p>❤ <b>{get_translation("lifespan", language)}</b>: {lines[5]}</p>
            <p>💉 <b>{get_translation("temperament", language)}</b>: {lines[6]}</p>
            <p>🥩 <b>{get_translation("productivity_metrics", language)}</b>: {lines[7]}</p>
        </div>
        """
        st.markdown(info_html, unsafe_allow_html=True)

        measurements = breed_data["measurements"]
        st.markdown(f"""
        <div style="background-color: rgba(232, 244, 248, 0.95); padding: 20px; border-radius: 10px; border-left: 5px solid #e74c3c; margin: 15px 0;">
            <h4>📏 {get_translation("physical_measurements", language)}</h4>
            <p>📏 <b>{get_translation("body_length", language)}</b>: {measurements['body_length']}</p>
            <p>📐 <b>{get_translation("height_withers", language)}</b>: {measurements['height_withers']}</p>
            <p>📊 <b>{get_translation("chest_width", language)}</b>: {measurements['chest_width']}</p>
            <p>📐 <b>{get_translation("rump_angle", language)}</b>: {measurements['rump_angle']}</p>
        </div>
        """, unsafe_allow_html=True)
    except:
        pass

# ============ CHATBOT ============
def chatbot_response(message):
    message = message.lower()
    if any(w in message for w in ["hello", "hi", "hey"]):
        return "Hello! How can I help you with cattle-related questions today?"
    elif "breed" in message or "identify" in message:
        return "Upload an image of cattle to identify its breed using our AI model. We can identify over 40 Indian cattle breeds!"
    elif any(w in message for w in ["buy", "sell", "market"]):
        return "Visit our Cattle Marketplace to buy or sell cattle. Click the 'Cattle Marketplace' button!"
    elif any(w in message for w in ["health", "sick", "disease"]):
        return "For health issues, consult a veterinarian. Common concerns include foot-and-mouth disease and mastitis."
    elif any(w in message for w in ["feed", "food", "diet"]):
        return "Cattle need balanced feed with proteins, energy, vitamins, and minerals. Common feeds include green fodder and concentrates."
    elif "milk" in message:
        return "Milk production varies by breed. Holstein Friesian can produce 20-30 liters/day, while indigenous breeds produce 10-15 liters/day."
    else:
        return "I'm here to help with cattle questions. Ask me about breeds, buying/selling, health, feeding, or general care."

# ============ SESSION STATE ============
if 'chat_open' not in st.session_state:
    st.session_state.chat_open = False
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with cattle-related questions?"}]
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"

def toggle_chat():
    st.session_state.chat_open = not st.session_state.chat_open

def navigate_to(page):
    st.session_state.current_page = page

# ============ SIDEBAR ============
with st.sidebar:
    st.header("📊 Classification History")
    csv_file = "cattle_classification_data.csv"
    if os.path.isfile(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if not df.empty:
                st.dataframe(df.tail(5), use_container_width=True)
                with open(csv_file, "rb") as file:
                    st.download_button("📥 Download CSV", data=file, file_name="cattle_classification_data.csv", mime="text/csv")
        except:
            st.info("No history available")
    else:
        st.info("No history available")
    
    st.markdown("---")
    st.info("🐄 Indian Cattle Breed Identifier\n\nIdentifies over 40 Indian cattle breeds using AI.")

# ============ MAIN APP ============
if st.session_state.current_page == "main":
    st.markdown(f'<h1 class="main-header">{get_translation("title", language)}</h1>', unsafe_allow_html=True)
    st.markdown(f'<h2 class="sub-header">{get_translation("subtitle", language)}</h2>', unsafe_allow_html=True)

    st.info(get_translation("upload_info", language))

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(get_translation("marketplace_button", language), use_container_width=True):
            navigate_to("marketplace")

    uploaded_file = st.file_uploader(get_translation("upload_label", language), type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="📷 Uploaded Image", use_container_width=True)

            with st.spinner(get_translation("analyzing", language)):
                if model is not None:
                    breed, confidence = predict_breed(image)
                else:
                    breed, confidence = demo_predict(image)
                    st.info(get_translation("demo_mode", language))

            if breed:
                st.markdown(f"""
                <div class="prediction-box">
                    <p style="font-weight: bold; font-size: 1.5rem;">{get_translation("predicted_breed", language)} <b>{breed}</b></p>
                    <p style="font-weight: bold; font-size: 1.2rem; color: #3498db;">{get_translation("confidence", language)}: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_to_csv(breed, f"{confidence:.2f}%", uploaded_file.name, timestamp)

                breed_key = breed.lower().strip()
                if breed_key in breed_info_raw:
                    st.subheader(get_translation("breed_info", language))
                    display_breed_info(breed_key, breed_info_raw[breed_key], language)
                else:
                    st.warning(get_translation("no_info", language))
        except Exception as e:
            st.error(f"{get_translation('processing_error', language)}: {str(e)}")

    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
        <p>{get_translation("refresh", language)}</p>
        <p>{get_translation("heritage", language)}</p>
    </div>
    """, unsafe_allow_html=True)

# ============ MARKETPLACE ============
elif st.session_state.current_page == "marketplace":
    st.markdown(f'<h1 class="main-header">{get_translation("marketplace_title", language)}</h1>', unsafe_allow_html=True)
    
    if st.button(get_translation("back_button", language)):
        navigate_to("main")
    
    marketplace_data = [
        {"name": "Gir Cow", "price": "₹65,000", "seller": "Rajesh Farms", "contact": "+91 98765 43210", "location": "Ahmedabad, Gujarat", "age": "4 years", "milk_yield": "12-15 L/day"},
        {"name": "Murrah Buffalo", "price": "₹85,000", "seller": "Singh Dairy", "contact": "+91 97654 32109", "location": "Ludhiana, Punjab", "age": "5 years", "milk_yield": "8-10 L/day"},
        {"name": "Sahiwal Cow", "price": "₹55,000", "seller": "Green Fields", "contact": "+91 96543 21098", "location": "Hisar, Haryana", "age": "3 years", "milk_yield": "10-12 L/day"},
        {"name": "Jersey Cow", "price": "₹45,000", "seller": "Modern Dairy", "contact": "+91 95432 10987", "location": "Pune, Maharashtra", "age": "4 years", "milk_yield": "18-20 L/day"},
        {"name": "Holstein Friesian", "price": "₹75,000", "seller": "Elite Dairy Farms", "contact": "+91 93210 98765", "location": "Bangalore, Karnataka", "age": "3 years", "milk_yield": "22-25 L/day"}
    ]
    
    for cattle in marketplace_data:
        st.markdown(f"""
        <div class="cattle-card">
            <div class="cattle-name">{cattle['name']}</div>
            <div class="cattle-price">{get_translation("price", language)}: {cattle['price']}</div>
            <div class="seller-info">{get_translation("age", language)}: {cattle['age']}</div>
            <div class="seller-info">{get_translation("milk_yield", language)}: {cattle['milk_yield']}</div>
            <div class="seller-info">{get_translation("seller", language)}: {cattle['seller']}</div>
            <div class="seller-info">{get_translation("contact", language)}: {cattle['contact']}</div>
            <div class="seller-info">{get_translation("location", language)}: {cattle['location']}</div>
        </div>
        """, unsafe_allow_html=True)

# ============ CHAT TOGGLE ============
st.markdown('<div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">', unsafe_allow_html=True)
if st.button("💬", key="chat_toggle", help="Chat with us"):
    toggle_chat()
st.markdown('</div>', unsafe_allow_html=True)

# ============ CHAT INTERFACE ============
if st.session_state.chat_open:
    st.markdown(f"""
    <div style='position: fixed; bottom: 90px; right: 20px; width: 350px; height: 420px; 
                background-color: white; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); 
                z-index: 1000; display: flex; flex-direction: column; overflow: hidden;'>
        <div style='background-color: #3498db; color: white; padding: 15px; font-weight: bold; 
                    border-top-left-radius: 15px; border-top-right-radius: 15px;'>
            {get_translation("chat_title", language)} 💬
        </div>
        <div style='flex: 1; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px;'>
    """, unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div style='max-width: 80%; padding: 10px 15px; border-radius: 15px; background-color: #3498db; color: white; align-self: flex-end;'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='max-width: 80%; padding: 10px 15px; border-radius: 15px; background-color: #f1f1f1; align-self: flex-start;'>{message['content']}</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.container():
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.text_input("", key="chat_input", placeholder="Type your message...", label_visibility="collapsed")
        with col2:
            if st.button("Send", key="chat_send", use_container_width=True):
                if user_input.strip():
                    st.session_state.messages.append({"role": "user", "content": user_input})
                    st.session_state.messages.append({"role": "assistant", "content": chatbot_response(user_input)})
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
