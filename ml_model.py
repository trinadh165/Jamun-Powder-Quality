"""
Machine Learning Model for Jamun Powder Quality Prediction
Updated to XGBoost for higher accuracy.
This file is a drop-in replacement for the original Random Forest version.

- Keeps same class name: JamunQualityModel
- Keeps same function: predict_quality()
- Generates same output structure for compatibility
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib
import warnings
import os

warnings.filterwarnings("ignore")


class JamunQualityModel:
    def _init_(self, model_path="jamun_powder_ml_model.pkl"):
        self.pipeline = None
        self.model_path = model_path

        # Load model if exists
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                print("Loaded existing model:", self.model_path)
            except:
                print("Error loading saved model. Retraining required.")

    # --------------------------------------------------------
    # Synthetic dataset generator (similar pattern to your original project)
    # --------------------------------------------------------
    def generate_dataset(self, n=3500, seed=42):
        np.random.seed(seed)

        ripeness = np.random.choice(['unripe', 'semi_ripe', 'ripe'], n, p=[0.2, 0.3, 0.5])
        uniformity = np.random.choice([True, False], n, p=[0.7, 0.3])
        freeze_temp = np.random.uniform(-60, -10, n)
        freeze_time = np.random.uniform(5, 40, n)
        packaging = np.random.choice(['plastic_bag', 'opaque_bag', 'vacuum_pouch'], n)
        opaque = np.random.choice([True, False], n, p=[0.6, 0.4])
        airtight = np.random.choice([True, False], n, p=[0.7, 0.3])
        storage_temp = np.random.choice(['normal', 'cool', 'frozen'], n)
        humidity = np.random.uniform(0, 25, n)
        light = np.random.choice(['exposed', 'low', 'dark'], n)

        quality = []
        for i in range(n):
            if (
                ripeness[i] == 'ripe'
                and -55 <= freeze_temp[i] <= -35
                and 15 <= freeze_time[i] <= 25
                and packaging[i] == 'vacuum_pouch'
                and opaque[i] and airtight[i]
                and storage_temp[i] == 'frozen'
                and humidity[i] < 8
                and light[i] == 'dark'
            ):
                quality.append('excellent')
            elif ripeness[i] == 'ripe' and packaging[i] in ['opaque_bag', 'vacuum_pouch'] and humidity[i] < 12:
                quality.append('very_good')
            elif ripeness[i] in ['semi_ripe', 'ripe'] and freeze_temp[i] < -20:
                quality.append('good')
            else:
                quality.append('bad')

        df = pd.DataFrame({
            'ripeness': ripeness,
            'quality_uniformity': uniformity,
            'freezing_temperature': freeze_temp,
            'freezing_time': freeze_time,
            'packaging_material': packaging,
            'is_opaque': opaque,
            'is_airtight': airtight,
            'storage_temperature': storage_temp,
            'humidity_percent': humidity,
            'light_exposure': light,
            'quality': quality
        })

        return df

    # --------------------------------------------------------
    # Build preprocessing + XGBoost pipeline
    # --------------------------------------------------------
    def build_pipeline(self):
        categorical_cols = [
            'ripeness', 'packaging_material',
            'storage_temperature', 'light_exposure'
        ]

        numerical_cols = [
            'freezing_temperature', 'freezing_time',
            'humidity_percent'
        ]

        boolean_cols = [
            'quality_uniformity', 'is_opaque', 'is_airtight'
        ]

        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
            ("bool", "passthrough", boolean_cols)
        ])

        xgb_model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss"
        )

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", xgb_model)
        ])

        return pipeline

    # --------------------------------------------------------
    # Train model and save pkl
    # --------------------------------------------------------
    def train(self, verbose=True):
        print("Generating dataset...")
        df = self.generate_dataset()

        X = df.drop("quality", axis=1)
        y = df["quality"]

        pipeline = self.build_pipeline()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training XGBoost model...")
        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"XGBoost Accuracy: {acc * 100:.2f}%")

        joblib.dump(pipeline, self.model_path)
        print("Saved model:", self.model_path)

        self.pipeline = pipeline

    # --------------------------------------------------------
    # Predict function used by app.py (same structure)
    # --------------------------------------------------------
    def predict_quality(self, user_input):
        if self.pipeline is None:
            raise ValueError("Model not loaded. Train first or ensure pkl exists.")

        df = pd.DataFrame([user_input])
        probs = self.pipeline.predict_proba(df)[0]
        classes = self.pipeline.classes_

        idx = np.argmax(probs)
        return {
            "predicted_quality": classes[idx],
            "confidence": float(probs[idx])
        }


# --------------------------------------------------------
# For standalone training
# --------------------------------------------------------
if _name_ == "_main_":
    model = JamunQualityModel()
    model.train()
