# src/data_processor.py

def process_audience_demographics(audience_data: dict) -> dict:
    """
    Processes and standardizes aggregated audience demographic data for the matchmaker.
    """
    
    regions = audience_data.get('follower_regions', {})
    age_dist = audience_data.get('follower_age_dist', {})
    gender_dist = audience_data.get('follower_gender_dist', {})
    
    # 1. Top Region
    top_region = max(regions, key=regions.get, default='N/A')
    top_region_perc = regions.get(top_region, 0)
    
    # 2. Key Audience Age Group Percentage (18-35 segment)
    age_18_25 = age_dist.get('18-25', 0)
    age_26_35 = age_dist.get('26-35', 0)
    target_age_perc = age_18_25 + age_26_35

    # 3. Gender Skew
    female_perc = gender_dist.get('Female', 0)
    
    return {
        'audience_top_region': f"{top_region} ({top_region_perc:.1f}%)",
        'audience_18_35_perc': target_age_perc,
        'audience_female_perc': female_perc,
        'raw_age_dist': age_dist, # Kept for visual analytics
        'raw_gender_dist': gender_dist # Kept for visual analytics
    }

def create_final_demographic_vector(influencer_data: dict, audience_data: dict) -> dict:
    """
    Combines influencer and audience results into the final vector structure.
    """
    return {
        'influencer_id': influencer_data['influencer_id'],
        'influencer_gender': influencer_data['gender'],
        'influencer_age_group': influencer_data['age_group'],
        'demographics_vector': {
            **process_audience_demographics(audience_data)
        }
        # This vector is ready for brand filtering in the Matchmaker module (next task)
    }