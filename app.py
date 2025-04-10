
# Install packages if needed

# Import common libraries
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load datasets
farmer_df = pd.read_csv('./farmer_advisor_dataset.csv')
market_df = pd.read_csv('./market_researcher_dataset.csv')

# Connect to SQLite (memory for now, file-based later)
conn = sqlite3.connect('agriculture_memory.db')
cursor = conn.cursor()

# Create simple tables for memory
cursor.execute('''CREATE TABLE IF NOT EXISTS farmer_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    land_size REAL,
                    crop_preference TEXT,
                    financial_goal TEXT
                )''')

cursor.execute('''CREATE TABLE IF NOT EXISTS market_info (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    region TEXT,
                    crop TEXT,
                    price REAL,
                    demand_score REAL
                )''')

# Insert sample data (if needed)
farmer_df.to_sql('farmer_info', conn, if_exists='replace', index=False)
market_df.to_sql('market_info', conn, if_exists='replace', index=False)

print("Setup complete ✅")

import pandas as pd

# Load Farmer Advisor Dataset
farmer_df = pd.read_csv('./farmer_advisor_dataset.csv')

# FIX: Remove spaces from column headers
farmer_df.columns = farmer_df.columns.str.strip()

# Check columns
print(farmer_df.columns.tolist())

# 1. Install missing libraries if needed

# 2. Import libraries
import pandas as pd

# 3. Load datasets
farmer_df = pd.read_csv('./farmer_advisor_dataset.csv')
market_df = pd.read_csv('./market_researcher_dataset.csv')

# 4. Clean columns
farmer_df.columns = farmer_df.columns.str.strip()
market_df.columns = market_df.columns.str.strip()

# 5. Preview data
print(farmer_df.head())
print(market_df.head())

# Smart Crop Recommender

def recommend_crop(soil_ph, soil_moisture, temperature, rainfall):
    # Filter crops that match soil and weather conditions
    recommended = farmer_df[
        (farmer_df['Soil_pH'].between(soil_ph - 0.5, soil_ph + 0.5)) &
        (farmer_df['Soil_Moisture'].between(soil_moisture - 10, soil_moisture + 10)) &
        (farmer_df['Temperature_C'].between(temperature - 5, temperature + 5)) &
        (farmer_df['Rainfall_mm'].between(rainfall - 50, rainfall + 50))
    ]

    if not recommended.empty:
        print("✅ Recommended Crops based on your soil and climate:")
        print(recommended[['Crop_Type', 'Crop_Yield_ton', 'Sustainability_Score']])
    else:
        print("⚠️ No perfect matches found, suggest soil treatment first!")

# Example usage
recommend_crop(soil_ph=6.5, soil_moisture=30, temperature=25, rainfall=200)

# Fertilizer Optimizer

def suggest_fertilizer_usage(crop_type):
    # Find average fertilizer usage for this crop
    crop_data = farmer_df[farmer_df['Crop_Type'] == crop_type]

    if not crop_data.empty:
        avg_fertilizer = crop_data['Fertilizer_Usage_kg'].mean()
        avg_pesticide = crop_data['Pesticide_Usage_kg'].mean()
        print(f"🌱 For {crop_type}, use around {avg_fertilizer:.2f} kg fertilizer and {avg_pesticide:.2f} kg pesticide per acre for optimal yield.")
    else:
        print("⚠️ Crop type not found in database.")

# Example usage
suggest_fertilizer_usage('Soybean')

# Sustainability Score Prediction (ML)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Prepare data
X = farmer_df[['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm', 'Fertilizer_Usage_kg', 'Pesticide_Usage_kg']]
y = farmer_df['Sustainability_Score']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error
print(f"🧪 Model Test MSE: {mean_squared_error(y_test, predictions):.2f}")

# Predict Sustainability for a new farm
def predict_sustainability(soil_ph, soil_moisture, temp, rainfall, fertilizer, pesticide):
    input_features = [[soil_ph, soil_moisture, temp, rainfall, fertilizer, pesticide]]
    predicted_score = model.predict(input_features)
    print(f"🔮 Predicted Sustainability Score: {predicted_score[0]:.2f}")

# Example usage
predict_sustainability(6.5, 30, 26, 220, 120, 5)

# Install pandas if needed

# Import
import pandas as pd

# Load datasets
farmer_df = pd.read_csv('./farmer_advisor_dataset.csv')
market_df = pd.read_csv('./market_researcher_dataset.csv')

# Clean column names
farmer_df.columns = farmer_df.columns.str.strip()
market_df.columns = market_df.columns.str.strip()

# Show to verify
print("Farmer dataset columns:", farmer_df.columns.tolist())
print("Market dataset columns:", market_df.columns.tolist())

# Farmer Class
class Farmer:
    def __init__(self, soil_ph, moisture, temp, rainfall):
        self.soil_ph = soil_ph
        self.moisture = moisture
        self.temp = temp
        self.rainfall = rainfall

    def analyze_conditions(self):
        print("\nAnalyzing soil and climate conditions...\n")
        recommended = farmer_df[
            (farmer_df['Soil_pH'].between(self.soil_ph - 0.5, self.soil_ph + 0.5)) &
            (farmer_df['Soil_Moisture'].between(self.moisture - 5, self.moisture + 5)) &
            (farmer_df['Temperature_C'].between(self.temp - 2, self.temp + 2)) &
            (farmer_df['Rainfall_mm'].between(self.rainfall - 50, self.rainfall + 50))
        ]
        print("✅ Recommended Crops based on your soil and climate:")
        print(recommended[['Crop_Type', 'Crop_Yield_ton', 'Sustainability_Score']])
        return recommended[['Crop_Type', 'Crop_Yield_ton', 'Sustainability_Score']]

# Market Researcher Class
class MarketResearcher:
    def suggest_profitable_crop(self):
        print("\nChecking profitable crops from market data...\n")
        # Correct columns: 'Market_Price_per_ton' and 'Demand_Index'
        best_crops = market_df.sort_values(
            ['Market_Price_per_ton', 'Demand_Index'],
            ascending=[False, False]
        ).head(5)
        print("✅ Top profitable crops in the market:")
        print(best_crops[['Product', 'Market_Price_per_ton', 'Demand_Index']])
        return best_crops[['Product', 'Market_Price_per_ton', 'Demand_Index']]

# Agricultural Advisor Class
class AgriculturalAdvisor:
    def give_recommendations(self, farmer, market_researcher):
        soil_crops = farmer.analyze_conditions()
        market_crops = market_researcher.suggest_profitable_crop()

        # Find common crops between soil recommendation and market demand
        common_crops = set(soil_crops['Crop_Type']).intersection(set(market_crops['Product']))

        if common_crops:
            print("\n✅ Final recommended crops (good soil + profitable market):")
            for crop in common_crops:
                print(f"👉 {crop}")
        else:
            print("\n⚠️ No common crops found between soil conditions and market profitability.")

# --- Main program ---

# Instantiate agents
farmer = Farmer(soil_ph=6.5, moisture=30, temp=25, rainfall=200)
market_researcher = MarketResearcher()
advisor = AgriculturalAdvisor()

# Give recommendations
advisor.give_recommendations(farmer, market_researcher)

class FertilizerAdvisor:
    def suggest_usage(self, crop_type):
        avg_fertilizer = farmer_df[farmer_df['Crop_Type'] == crop_type]['Fertilizer_Usage_kg'].mean()
        avg_pesticide = farmer_df[farmer_df['Crop_Type'] == crop_type]['Pesticide_Usage_kg'].mean()
        print(f"🌿 For {crop_type}:")
        print(f"Recommended Fertilizer Usage: {avg_fertilizer:.2f} kg/acre")
        print(f"Recommended Pesticide Usage: {avg_pesticide:.2f} kg/acre")

fertilizer_advisor = FertilizerAdvisor()
fertilizer_advisor.suggest_usage('Corn')

from sklearn.linear_model import LinearRegression

class YieldPredictor:
    def __init__(self):
        features = ['Soil_pH', 'Soil_Moisture', 'Temperature_C', 'Rainfall_mm']
        X = farmer_df[features]
        y = farmer_df['Crop_Yield_ton']
        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict_yield(self, soil_ph, moisture, temp, rainfall):
        input_data = [[soil_ph, moisture, temp, rainfall]]
        predicted_yield = self.model.predict(input_data)[0]
        print(f"🌾 Predicted Crop Yield: {predicted_yield:.2f} tons per acre")

yield_predictor = YieldPredictor()
yield_predictor.predict_yield(soil_ph=6.5, moisture=30, temp=25, rainfall=100)

class SustainabilityAdvisor:
    def recommend_sustainable_crops(self):
        sustainable = farmer_df[farmer_df['Sustainability_Score'] > 70]
        print("♻️ Sustainable Crops to Focus On:")
        print(sustainable[['Crop_Type', 'Crop_Yield_ton', 'Sustainability_Score']].drop_duplicates())

sustainability_advisor = SustainabilityAdvisor()
sustainability_advisor.recommend_sustainable_crops()

class MarketTrendAnalyzer:
    def top_trending_crops(self):
        trending = market_df.sort_values('Demand_Index', ascending=False).head(5)
        print("📈 Top Trending Crops:")
        print(trending[['Product', 'Demand_Index', 'Market_Price_per_ton']])


trend_analyzer = MarketTrendAnalyzer()
trend_analyzer.top_trending_crops()

def interactive_advisor():
    soil_ph = float(input("Enter Soil pH (e.g., 6.5): "))
    moisture = float(input("Enter Soil Moisture (%): "))
    temp = float(input("Enter Temperature (°C): "))
    rainfall = float(input("Enter Rainfall (mm): "))

    farmer = Farmer(soil_ph, moisture, temp, rainfall)
    recommendations = farmer.analyze_conditions()

    if not recommendations.empty:
        print("\n✅ Based on your conditions, recommended crops are:")
        print(recommendations['Crop_Type'].unique())
    else:
        print("⚠️ No exact match found, consider adjusting farming parameters.")

from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Load your processed data
recommended_crops = ['Wheat', 'Rice']
trending_crops = [
    {"Product": "Corn", "Demand_Index": 199.99, "Market_Price_per_ton": 454.79},
    {"Product": "Soybean", "Demand_Index": 199.98, "Market_Price_per_ton": 141.79},
]
sustainable_crops = [
    {"Crop_Type": "Wheat", "Crop_Yield_ton": 8.87, "Sustainability_Score": 89.76},
    {"Crop_Type": "Soybean", "Crop_Yield_ton": 6.22, "Sustainability_Score": 82.92},
]
prediction_summary = {
    "sustainability_score": 50.80,
    "predicted_yield": 5.50
}

@app.route('/api/recommendations')
def get_recommendations():
    return jsonify(recommended_crops)

@app.route('/api/trending')
def get_trending():
    return jsonify(trending_crops)

@app.route('/api/sustainable')
def get_sustainable():
    return jsonify(sustainable_crops)

@app.route('/api/predictions')
def get_predictions():
    return jsonify(prediction_summary)

if __name__ == '__main__':
    app.run(debug=True)
