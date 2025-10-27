# Member 5: Gender & Demographic Classifier Module

This module is responsible for analyzing an influencer’s profile (Image and Bio) and aggregating their audience data into a standardized **Demographic Feature Vector**.  
This vector is a crucial input for the final Matchmaker module.

---

## 1. Installation and Setup 🛠️

### 1.1. Cloning the Repository

```bash
git clone [YOUR_REPO_URL]
cd demographic-classifier-module
```

### 1.2. Virtual Environment

It is **required** to use a virtual environment (`venv`) to manage the complex dependencies (especially DeepFace/TensorFlow).

```bash
# Create and activate the environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate.bat # Windows
```

### 1.3. Install Dependencies

Install all necessary packages, including the large ML libraries (DeepFace, Torch, Transformers):

```bash
pip install -r requirements.txt
```

### 1.4. Required Asset (Image)

For the Computer Vision (CV) component to work, you must provide the profile picture.

- **Asset**: Place the influencer’s profile image in the root directory.
- **Filename**: Name the file exactly: `test_image.jpg`
- **Path Fix**: If you get a “file not found” error, you must update the `IMAGE_FILE_PATH` variable in `main.py` to use the full, absolute path on your system.

---

## 2. Running the Module (Local Test) 🚀

To test the module and generate a sample output vector, run the main script from the root directory:

```bash
python main.py
```

**Output**  
The console will display the final vector, and it will be saved to a file:

```
demographics_output.json
```

---

## 3. Integration API (The Contract) 🤝

The final Matchmaker module must **import** and **call** the primary processing function, feeding it the required raw data.

### 3.1. Function to Import

The Matchmaker should import the core feature-creation function:

```python
from demographic_classifier_module.src.data_processor import create_final_demographic_vector
```

---

### 3.2. Data Inputs (Required from Scraper/Data Member)

The integration module must pass a dictionary containing the raw audience data.  
**This dictionary must follow the exact structure** used in our mock data:

| Key | Type | Description | Example Structure |
|-----|------|--------------|-------------------|
| `follower_regions` | dict | Percentage breakdown of follower countries. | `{"US": 40.2, "CAN": 15.1, ...}` |
| `follower_age_dist` | dict | Percentage breakdown of age bins. | `{"18-25": 20.0, "26-35": 45.0, ...}` |
| `follower_gender_dist` | dict | Percentage breakdown of follower genders. | `{"Female": 19.5, "Male": 80.5}` |

---

### 3.3. Output Feature Vector (Provided to Matchmaker)

The function returns the final feature vector with the following core features:

| Key | Description | Relevance (Feature for Matching) |
|-----|--------------|----------------------------------|
| `influencer_gender` | Classified gender (MALE/FEMALE) from the ensemble (Image + Bio). | For brands targeting the influencer’s own gender. |
| `influencer_age_group` | Classified age group (e.g., 26–35) from the profile image. | For brands targeting the influencer’s life stage. |
| `audience_top_region` | The primary region where the audience is located (e.g., “US 40.2%”). | For geo-specific campaigns. |
| `audience_18_35_perc` | Total percentage of followers aged **18 to 35**. | For matching brands seeking the key young adult consumer segment. |
| `audience_female_perc` | Percentage of female followers. | For matching gender-specific products. |

---

## 4. Raw Age and Gender Data Explanation 📊

### What is `raw_age_group` and `gender_dist`?

The parameters `raw_age_dist` and `raw_gender_dist` are simply the **unprocessed, complete demographic data** gathered from the platform’s analytics.

They are included in the final vector not as core matching features, but as **auxiliary data** for visualization and deep-dive analysis.

---

### Raw Age and Gender Distributions Explained

| Parameter | Meaning | Format | Purpose in the Vector |
|------------|----------|---------|------------------------|
| `raw_age_dist` | Raw Age Distribution | A dictionary mapping every available age bracket to the percentage of followers within that bracket. | Provides the full picture of the audience’s age spread (useful for charts or brand-specific targeting). |
| `raw_gender_dist` | Raw Gender Distribution | A dictionary mapping every available gender category to the percentage of followers. | Provides full raw percentages (should sum to 100%), confirming platform analytics accuracy. |

---

📦 **Export to Sheets:** Each generated dictionary can optionally be exported to Google Sheets for reporting or analysis.
