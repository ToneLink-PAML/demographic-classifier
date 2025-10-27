# main.py

import json
import os # Import the os module for path manipulation
from src.demographic_classifier import get_influencer_demographics
from src.data_processor import create_final_demographic_vector

# --- Setup ---
# ⚠️ ACTION REQUIRED: REPLACE THIS PATH with the full, absolute path on your system.
# Example: '/Users/rachnaissar/Documents/Code/demographics-classifier/test_image.jpg'
IMAGE_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_image.jpg')
# If the above fails, use the hardcoded ABSOLUTE path:
# IMAGE_FILE_PATH = '/YOUR/FULL/ABSOLUTE/PATH/TO/demographics-classifier/test_image.jpg' 

# --- Mock Data ---
MOCK_INFLUENCER_DATA = {
    'id': 'A456_Wellness',
    'image_path': IMAGE_FILE_PATH, 
    # MALE bio text (assumes this is the desired output)
    'bio_text': 'Dad, gearhead, and fitness enthusiast. Sharing my weightlifting routines and favorite tools. #hehim #gymlife',
    'audience_raw': {
        # Audience metrics aligned with MALE influencer
        'follower_regions': {'US': 40.2, 'CA': 15.1, 'UK': 10.5, 'DE': 5.0, 'Other': 29.2},
        'follower_age_dist': {'<18': 5.0, '18-25': 20.0, '26-35': 45.0, '36-50': 25.0, '50+': 5.0},
        'follower_gender_dist': {'Female': 19.5, 'Male': 80.5}
    }
}

# --- REMAINDER OF run_member5_classifier function (omitted for brevity) ---
def run_member5_classifier():
    """Runs the full Member 5 pipeline."""
    print("--- 👤 Member 5: Gender & Demographic Classifier ---")
    
    # Check if the image path is valid before attempting DeepFace
    if not os.path.exists(MOCK_INFLUENCER_DATA['image_path']):
        print(f"FATAL ERROR: Image not found at the resolved path: {MOCK_INFLUENCER_DATA['image_path']}")
        print("Please ensure 'test_image.jpg' exists and update IMAGE_FILE_PATH.")
        
    # 1. Classify Influencer Demographics (CV + NLP)
    influencer_res = get_influencer_demographics(
        MOCK_INFLUENCER_DATA['id'],
        MOCK_INFLUENCER_DATA['image_path'],
        MOCK_INFLUENCER_DATA['bio_text']
    )
    
    # 2. Process and Combine with Audience Data
    final_vector = create_final_demographic_vector(
        influencer_res,
        MOCK_INFLUENCER_DATA['audience_raw']
    )
    
    # 3. Output Validation (F1-score/Accuracy would be calculated during model training)
    
    # 4. Store/Display Results
    print("\n✅ Final Demographic Vector Generated:")
    
    # Display influencer-level data
    print(f"   Influencer ID: {final_vector['influencer_id']}")
    print(f"   Inferred Gender: {final_vector['influencer_gender']}")
    print(f"   Inferred Age Group: {final_vector['influencer_age_group']}")
    
    # Display audience-level data
    dem_vec = final_vector['demographics_vector']
    print("\n   --- Audience Targeting Metrics ---")
    print(f"   Audience Top Region: {dem_vec['audience_top_region']}")
    print(f"   Audience 18-35% Share: {dem_vec['audience_18_35_perc']:.1f}%")
    print(f"   Audience Female Share: {dem_vec['audience_female_perc']:.1f}%")
    
    # Store result (as JSON for use by other modules)
    with open('demographics_output.json', 'w') as f:
        json.dump(final_vector, f, indent=4)
    print("\n💾 Results saved to demographics_output.json")

if __name__ == "__main__":
    run_member5_classifier()