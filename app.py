# rules.py (Conceptual implementation with XGBoost)

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib # For saving/loading the model

# --- Model Initialization and Training (Run once on startup/deployment) ---

# 1. MOCK DATA GENERATION (Replace with your actual data)
def generate_mock_data(n_samples=5000):
    np.random.seed(42)
    data = {}
    
    # Input Features
    ripeness_map = {'unripe': 0, 'optimal': 1, 'overripe': 2}
    packaging_map = {'plastic': 0, 'glass_jar': 1, 'vacuum_pouch': 2}
    storage_map = {'room': 0, 'refrigerated': 1, 'frozen': 2}
    exposure_map = {'high': 0, 'medium': 1, 'low': 2}
    
    data['ripeness'] = np.random.choice(list(ripeness_map.keys()), n_samples, p=[0.1, 0.7, 0.2])
    data['quality_uniformity'] = np.random.choice([True, False], n_samples, p=[0.85, 0.15])
    data['temperature'] = np.random.uniform(-40, -10, n_samples) # Freeze-drying temp (C)
    data['time_hours'] = np.random.uniform(20, 48, n_samples)
    data['packaging_type'] = np.random.choice(list(packaging_map.keys()), n_samples, p=[0.4, 0.3, 0.3])
    data['opaque'] = np.random.choice([True, False], n_samples, p=[0.7, 0.3])
    data['airtight'] = np.random.choice([True, False], n_samples, p=[0.8, 0.2])
    data['storage_temperature'] = np.random.choice(list(storage_map.keys()), n_samples, p=[0.1, 0.4, 0.5])
    data['humidity_percent'] = np.random.uniform(5, 75, n_samples)
    data['light_exposure'] = np.random.choice(list(exposure_map.keys()), n_samples, p=[0.15, 0.35, 0.5])
    
    df = pd.DataFrame(data)
    
    # Target Variable (Quality Score 0-4: Poor, Fair, Good, Very Good, Excellent)
    # This mock formula biases towards optimal ripeness, low temp/time, vacuum, frozen storage
    score_influence = (
        df['ripeness'].map(ripeness_map) * 0.5 + 
        df['quality_uniformity'] * 1.5 + 
        (df['temperature'] + 40) / 30 * 0.8 + # Lower temp is better
        (48 - df['time_hours']) / 28 * 0.6 +  # Shorter time is better
        df['packaging_type'].map(packaging_map) * 1.0 +
        df['opaque'] * 0.5 +
        df['airtight'] * 1.0 +
        df['storage_temperature'].map(storage_map) * 1.5 -
        df['humidity_percent'] / 75 * 1.5 -
        df['light_exposure'].map(exposure_map) * 0.5
    )
    
    # Scale and discretize to 5 classes (0 to 4)
    min_score = score_influence.min()
    max_score = score_influence.max()
    normalized_score = (score_influence - min_score) / (max_score - min_score)
    df['quality_class'] = pd.cut(normalized_score, bins=5, labels=False, include_lowest=True).astype(int)
    
    return df

# 2. FEATURE ENGINEERING & TRAINING
def train_and_load_model():
    df = generate_mock_data()
    
    # Separate features (X) and target (y)
    X = df.drop('quality_class', axis=1)
    y = df['quality_class']
    
    # One-Hot Encode Categorical Features
    X = pd.get_dummies(X, columns=['ripeness', 'packaging_type', 'storage_temperature', 'light_exposure'], drop_first=True)
    
    # Rename for compatibility with XGBoost (no special chars)
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize and train the XGBoost model
    # Use 'multi:softmax' for multi-class classification
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=5,
        n_estimators=300,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"XGBoost Model Trained with Accuracy: {acc:.4f}")
    
    # Save model and feature names for later use in Flask app
    joblib.dump(model, 'jamun_quality_model.pkl')
    joblib.dump(X_train.columns.tolist(), 'model_features.pkl')

    return model, X.columns.tolist(), X_test, y_test

# Try to load the model, otherwise train it
MODEL_PATH = 'jamun_quality_model.pkl'
FEATURES_PATH = 'model_features.pkl'

try:
    xgb_model = joblib.load(MODEL_PATH)
    model_features = joblib.load(FEATURES_PATH)
    # Mock test set for metric generation (In a real app, this would be loaded)
    _, _, X_test_mock, y_test_mock = train_and_load_model() 
    print("XGBoost Model Loaded Successfully.")
except FileNotFoundError:
    print("Model not found. Training new XGBoost model...")
    xgb_model, model_features, X_test_mock, y_test_mock = train_and_load_model()


# --- Feature Preprocessing Function ---

def preprocess_input(data, feature_names):
    """Converts raw dictionary input from the Flask form into a DataFrame for XGBoost."""
    
    # 1. Flatten the input structure into a single dictionary
    flat_data = {
        'ripeness': data['raw_material']['ripeness'],
        'quality_uniformity': data['raw_material']['quality_uniformity'],
        'temperature': data['freeze_drying']['temperature'],
        'time_hours': data['freeze_drying']['time_hours'],
        'packaging_type': data['packaging']['type'],
        'opaque': data['packaging']['opaque'],
        'airtight': data['packaging']['airtight'],
        'storage_temperature': data['storage']['temperature'],
        'humidity_percent': data['storage']['humidity_percent'],
        'light_exposure': data['storage']['light_exposure']
    }
    
    # 2. Convert to DataFrame
    input_df = pd.DataFrame([flat_data])
    
    # 3. One-Hot Encode Categorical Features
    input_df = pd.get_dummies(input_df, 
                              columns=['ripeness', 'packaging_type', 'storage_temperature', 'light_exposure'], 
                              drop_first=True)
                              
    # 4. Align columns with the trained model's features
    # Ensure all original features are present, filling missing with 0 (standard for OHE)
    final_features = pd.DataFrame(0, index=input_df.index, columns=feature_names)
    for col in input_df.columns:
        if col in final_features.columns:
            final_features[col] = input_df[col]

    return final_features


# --- Evaluation Function (Called by app.py) ---

def evaluate_jamun_powder_quality(data):
    """
    Evaluates Jamun powder quality using the trained XGBoost model.
    The output is structured to match the original app's expected output.
    """
    
    # 1. Preprocess input
    input_features = preprocess_input(data, model_features)
    
    # 2. Get quality prediction (0=Poor, 4=Excellent)
    quality_class_int = xgb_model.predict(input_features)[0]
    
    # Map the integer prediction back to a readable class/score
    quality_map = {
        0: 'Poor', 1: 'Fair', 2: 'Good', 3: 'Very Good', 4: 'Excellent'
    }
    
    quality_label = quality_map[quality_class_int]
    
    # 3. Create dummy/rule-based secondary results based on the main prediction
    # Since the full model for all sub-metrics (color, flow, shelf life) is 
    # not provided, we must infer or mock them based on the main quality prediction.
    
    if quality_class_int == 4: # Excellent
        scores = {'vc': 95, 'anti': 98, 'color': 'Deep Purple', 'flow': 'Excellent', 'stab': 'High', 'shelf': 12}
    elif quality_class_int == 3: # Very Good
        scores = {'vc': 85, 'anti': 90, 'color': 'Purple', 'flow': 'Good', 'stab': 'Moderate', 'shelf': 9}
    elif quality_class_int == 2: # Good
        scores = {'vc': 70, 'anti': 75, 'color': 'Faded Purple', 'flow': 'Fair', 'stab': 'Average', 'shelf': 6}
    elif quality_class_int == 1: # Fair
        scores = {'vc': 50, 'anti': 55, 'color': 'Brownish', 'flow': 'Poor', 'stab': 'Low', 'shelf': 3}
    else: # Poor
        scores = {'vc': 30, 'anti': 35, 'color': 'Light Brown', 'flow': 'Very Poor', 'stab': 'Very Low', 'shelf': 1}
        
    return {
        "overall_quality": quality_label,
        "nutrition": {
            "vitamin_c_score": scores['vc'],
            "antioxidant_score": scores['anti']
        },
        "physical": {
            "color_quality": scores['color'],
            "powder_flow": scores['flow']
        },
        "chemical": {
            "moisture_stability": scores['stab']
        },
        "shelf_life": {
            "estimated_months": scores['shelf']
        }
    }

# --- Metrics Functions (Called by app.py for /metrics route) ---

def get_model_performance_metrics():
    """Returns model performance metrics (Accuracy, Precision, Recall, F1)"""
    y_pred = xgb_model.predict(X_test_mock)
    
    accuracy = accuracy_score(y_test_mock, y_pred) * 100
    # Use 'micro' or 'weighted' for multi-class metrics
    precision = precision_score(y_test_mock, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test_mock, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test_mock, y_pred, average='weighted', zero_division=0) * 100
    
    return {
        'accuracy': round(accuracy, 2),
        'precision': round(precision, 2),
        'recall': round(recall, 2),
        'f1_score': round(f1, 2),
        'cv_mean': 98.42,  # Keep fixed/mock for simplicity
        'cv_std': 0.60,    # Keep fixed/mock for simplicity
        'training_samples': X_test_mock.shape[0] * 4, # Mock
        'test_samples': X_test_mock.shape[0], 
        'features': len(model_features)
    }

def get_feature_importance_data():
    """Returns a list of feature importance dictionaries for Plotly graph"""
    importance = xgb_model.get_booster().get_score(importance_type='gain')
    total_gain = sum(importance.values())
    
    feature_importance_list = []
    
    for feature, gain in importance.items():
        feature_importance_list.append({
            'feature': feature,
            'importance': round((gain / total_gain) * 100, 2)
        })

    # Sort by importance
    feature_importance_list.sort(key=lambda x: x['importance'], reverse=True)
    
    # Map raw feature names back to readable names if necessary (e.g., ripeness_optimal -> Optimal Ripeness)
    # Simplified mapping for this example:
    readable_map = {
        'quality_uniformity': 'Quality Uniformity',
        'temperature': 'Freeze-Drying Temperature',
        'time_hours': 'Freeze-Drying Time (Hours)',
        'opaque': 'Opaque Packaging',
        'airtight': 'Airtight Packaging',
        'humidity_percent': 'Humidity Percent',
        'ripeness_optimal': 'Optimal Ripeness',
        'storage_temperature_frozen': 'Frozen Storage',
        'packaging_type_vacuum_pouch': 'Vacuum Pouch',
        # ... add others as needed
    }
    
    for item in feature_importance_list:
        item['feature'] = readable_map.get(item['feature'], item['feature'].replace('_', ' ').title())

    return feature_importance_list

# NOTE: You would also need a 'utils.py' with the 'validate_input_structure' function.
