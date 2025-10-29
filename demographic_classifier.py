import os
import re
from deepface import DeepFace
import torch
from transformers import pipeline

# --- Configuration ---
# MODEL_NAME is set to a common DistilBERT model. 
# In a final project, this would be replaced with a model specifically fine-tuned for gender/identity classification.
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" 

# Set device for PyTorch/Hugging Face
device = 0 if torch.cuda.is_available() else -1

# Initialize the NLP pipeline globally to avoid slow loading on every function call
# We initialize it here, but it only loads weights on the first use.
try:
    # Use a basic classification model as a stand-in for a custom gender classifier
    NLP_CLASSIFIER = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)
    print(f"INFO: Transformers pipeline loaded using {MODEL_NAME}.")
except Exception as e:
    # Fallback to None if the library/model fails to load
    print(f"WARNING: Could not load Transformers model. Falling back to rule-based NLP. Error: {e}")
    NLP_CLASSIFIER = None


def classify_from_image(image_path: str) -> dict:
    """
    Classifies gender and age group from a profile image using DeepFace (CV).
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found at {image_path}. Age and Image Gender defaulting to UNKNOWN.")
        return {'gender': 'UNKNOWN', 'age_group': 'UNKNOWN', 'confidence': 0.0}

    try:
        # DeepFace analyzes age and gender. enforce_detection=False allows analysis 
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

        # 2. Gender and Confidence
        gender_prediction = result.get('gender', {})
        gender = max(gender_prediction, key=gender_prediction.get, default='UNKNOWN')
        confidence = gender_prediction.get(gender, 0.0) / 100.0

        return {
            'gender': gender.upper(),
            'age_group': age_group,
            'confidence': confidence
        }

    except Exception as e:
        print(f"DeepFace analysis failed for {image_path}: {e}")
        return {'gender': 'UNKNOWN', 'age_group': 'UNKNOWN', 'confidence': 0.0}


def classify_from_bio(bio_text: str) -> dict:
    """
    Classifies gender from influencer bio text using a Hugging Face model or a rule-based fallback.
    """
    cleaned_text = re.sub(r'#|@|http\S+|pic\.\S+', '', bio_text).strip()
    if not cleaned_text:
        return {'gender': 'UNKNOWN', 'confidence': 0.0}

    # --- 1. BERT/Transformer Usage ---
    if NLP_CLASSIFIER:
        try:
            # NOTE: Since the model is a sentiment model, we use rule-based mapping 
            # to simulate a gender prediction based on common gender cues in text.
            result = NLP_CLASSIFIER(cleaned_text)[0]
            
            # SIMULATED GENDER MAPPING based on bio content for demonstration
            bio_text_lower = cleaned_text.lower()
            
            if any(word in bio_text_lower for word in ['she/her', 'mama', 'girl boss']):
                 # If these cues are present, assign FEMALE with high confidence
                return {'gender': 'FEMALE', 'confidence': 0.95} 
            
            if any(word in bio_text_lower for word in ['he/him', 'dad', 'guy', 'gearhead']):
                 # If these cues are present, assign MALE with high confidence
                return {'gender': 'MALE', 'confidence': 0.95}

            # If no strong gender cues, but the transformer ran successfully
            # This is where the actual fine-tuned model's output would be used.
            return {'gender': 'UNKNOWN', 'confidence': 0.5} 

        except Exception as e:
            print(f"NLP classification failed: {e}. Falling back to basic rule check.")
            
    # --- 2. Fallback (Rule-based stand-in) ---
    bio_text_lower = cleaned_text.lower()
    
    if any(word in bio_text_lower for word in ['she', 'her', 'woman']):
        return {'gender': 'FEMALE', 'confidence': 0.80}
    
    if any(word in bio_text_lower for word in ['he', 'him', 'man']):
        return {'gender': 'MALE', 'confidence': 0.80}
        
    return {'gender': 'UNKNOWN', 'confidence': 0.0}


def ensemble_gender_prediction(image_result: dict, bio_result: dict) -> str:
    """
    Combines Image and Bio predictions with weighting.
    (Logic remains the same, but now uses the output of the active pipeline.)
    """
    img_g, img_c = image_result['gender'], image_result['confidence']
    bio_g, bio_c = bio_result['gender'], bio_result['confidence']
    
    # 1. Direct Agreement/Trivial Cases
    if img_g == bio_g and img_g != 'UNKNOWN':
        return img_g
    
    if img_g == 'UNKNOWN' and bio_g == 'UNKNOWN':
        return 'UNKNOWN'

    # 2. Weighted Disagreement/Partial Info
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

# ... (get_influencer_demographics function is omitted as it is unchanged)
def get_influencer_demographics(influencer_id: str, image_path: str, bio_text: str) -> dict:
    """Runs all classification models for the influencer and returns the result."""
    img_res = classify_from_image(image_path)
    bio_res = classify_from_bio(bio_text)
    
    final_gender = ensemble_gender_prediction(img_res, bio_res)
    final_age_group = img_res['age_group'] 
    
    return {
        'influencer_id': influencer_id,
        'gender': final_gender,
        'age_group': final_age_group
    }
