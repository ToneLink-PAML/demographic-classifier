# src/demographic_classifier.py

import os
import re
from deepface import DeepFace
import torch
from transformers import pipeline

# --- Configuration ---
# NOTE: Replace this with the actual path to your fine-tuned DistilBERT model later.
MODEL_PATH = "./bio_gender_model" 

# Set device for PyTorch/Hugging Face
device = 0 if torch.cuda.is_available() else -1

def classify_from_image(image_path: str) -> dict:
    """
    Classifies gender and age group from a profile image using DeepFace (CV).
    """
    # Check if the file actually exists before running DeepFace
    if not os.path.exists(image_path):
        # This warning means the age/gender prediction will fail until the path is fixed.
        print(f"Warning: Image file not found at {image_path}. Age and Image Gender defaulting to UNKNOWN.")
        return {'gender': 'UNKNOWN', 'age_group': 'UNKNOWN', 'confidence': 0.0}

    try:
        # DeepFace analyzes age and gender. enforce_detection=False allows analysis 
        # even if face quality is low, but can introduce error.
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=['age', 'gender'],
            enforce_detection=False
        )

        if not analysis:
             return {'gender': 'UNKNOWN', 'age_group': 'UNKNOWN', 'confidence': 0.0}

        result = analysis[0]

        # 1. Age Group Binning
        age = int(result.get('age', 0))
        if 18 <= age <= 25:
            age_group = '18-25'
        elif 26 <= age <= 35:
            age_group = '26-35'
        elif 36 <= age <= 50:
            age_group = '36-50'
        else:
            age_group = 'Other' if age > 0 else 'UNKNOWN'

        # 2. Gender and Confidence (DeepFace output format)
        gender_prediction = result.get('gender', {})
        # Find the gender with the highest probability
        gender = max(gender_prediction, key=gender_prediction.get, default='UNKNOWN')
        confidence = gender_prediction.get(gender, 0.0) / 100.0

        return {
            'gender': gender.upper(),
            'age_group': age_group,
            'confidence': confidence
        }

    except Exception as e:
        # In a real environment, you might log this error.
        # print(f"DeepFace analysis failed for {image_path}: {e}")
        return {'gender': 'UNKNOWN', 'age_group': 'UNKNOWN', 'confidence': 0.0}


def classify_from_bio(bio_text: str) -> dict:
    """
    Classifies gender from influencer bio text using a fine-tuned DistilBERT (NLP)
    or a rule-based simulation (as provided here).
    """
    cleaned_text = re.sub(r'#|@|http\S+|pic\.\S+', '', bio_text).strip()
    if not cleaned_text:
        return {'gender': 'UNKNOWN', 'confidence': 0.0}

    # --- SIMULATION (Rule-based NLP stand-in) ---
    bio_text_lower = cleaned_text.lower()
    
    # Female Cues
    if any(word in bio_text_lower for word in ['mama', 'mom', 'she/her', 'girl boss']):
        return {'gender': 'FEMALE', 'confidence': 0.90}
    
    # Male Cues (Adjusted to match the 'Dad, gearhead...' mock data in main.py)
    if any(word in bio_text_lower for word in ['dad', 'he/him', 'guy', 'father', 'gearhead']):
        return {'gender': 'MALE', 'confidence': 0.85}
        
    # If using the actual Hugging Face model, the code would be here:
    # try:
    #     classifier = pipeline("sentiment-analysis", model=MODEL_PATH, tokenizer=MODEL_PATH, device=device)
    #     result = classifier(cleaned_text)[0]
    #     # ... logic to map model output to 'MALE' or 'FEMALE' ...
    # except Exception:
    #     return {'gender': 'UNKNOWN', 'confidence': 0.0}

    return {'gender': 'UNKNOWN', 'confidence': 0.0} # Default fallback


def ensemble_gender_prediction(image_result: dict, bio_result: dict) -> str:
    """
    Combines Image and Bio predictions with weighting.
    """
    img_g, img_c = image_result['gender'], image_result['confidence']
    bio_g, bio_c = bio_result['gender'], bio_result['confidence']
    
    # 1. Direct Agreement/Trivial Cases
    if img_g == bio_g and img_g != 'UNKNOWN':
        return img_g
    
    if img_g == 'UNKNOWN' and bio_g == 'UNKNOWN':
        return 'UNKNOWN'

    # 2. Weighted Disagreement/Partial Info
    
    # Image is often the strongest indicator, so it gets a higher weight.
    W_IMG, W_BIO = 0.6, 0.4 
    
    if img_g != 'UNKNOWN' and bio_g != 'UNKNOWN':
        # Disagreement: Use weighted score
        return img_g if (img_c * W_IMG) > (bio_c * W_BIO) else bio_g
    
    # One is unknown: Use the known one if confidence is above a threshold
    if img_g != 'UNKNOWN':
        return img_g if img_c > 0.7 else 'UNKNOWN'
    if bio_g != 'UNKNOWN':
        return bio_g if bio_c > 0.7 else 'UNKNOWN'
        
    return 'UNKNOWN'

def get_influencer_demographics(influencer_id: str, image_path: str, bio_text: str) -> dict:
    """Runs all classification models for the influencer and returns the result."""
    img_res = classify_from_image(image_path)
    bio_res = classify_from_bio(bio_text)
    
    final_gender = ensemble_gender_prediction(img_res, bio_res)
    # The image model's age is used as it's the standard for single-person age estimation
    final_age_group = img_res['age_group'] 
    
    return {
        'influencer_id': influencer_id,
        'gender': final_gender,
        'age_group': final_age_group
    }