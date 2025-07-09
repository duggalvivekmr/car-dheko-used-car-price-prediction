# Generate a script that predicts and denormalizes the price using actual max price from training data
import pandas as pd
from price_predictor import predict_price

# Costants
MIN_PRICE = 100000
MAX_PRICE = 9600000

def inverse_price(predicted_value, min_price=100000, max_price=9600000):
    return predicted_value * (max_price - min_price) + min_price

# Load model_data to retrieve max actual price
df = pd.read_csv(r"D:\Education\Data Science\Project\car-dheko-used-car-price-prediction\data\processed\model_data.csv")
max_price_actual = df["new_car_detail_13_priceActual"].max()
print("🔢 Max actual price used for scaling:", max_price_actual)


# Example input
test_input = {
    "car_links": 0.312081,
    "city": "Bangalore",
    "new_car_detail_0_it": 0,
    "new_car_detail_1_ft": "Petrol",
    "new_car_detail_2_bt": "Hatchback",
    "new_car_detail_3_km": "11949",
    "new_car_detail_4_transmission": "Manual",
    "new_car_detail_5_ownerNo": 1,
    "new_car_detail_6_owner": "1st Owner",
    "new_car_detail_7_oem": "Tata",
    "new_car_detail_8_model": "Tata Tiago",
    "new_car_detail_9_modelYear": 2017,
    "new_car_detail_10_centralVariantId": 2985,
    "new_car_detail_11_variantName": "1.2 Revotron XZ WO Alloy",
    "new_car_detail_13_priceActual": 585000,
    "new_car_detail_14_priceSaving": 0,
    "new_car_detail_15_priceFixedText": 0,
    "car_age": 7,
    "new_car_overview_1_top_Year of Manufacture": 2017,
    "new_car_specs_1_top_Seats": 5
}

# Predict normalized price
normalized_price = predict_price(test_input)

# Denormalize
denormalized_price = inverse_price(normalized_price)

print(f"🧮 Normalized Prediction: {normalized_price:.4f}")
print(f"🪙 Estimated Price (₹): ₹{int(denormalized_price):,}")

from src.logger import log_prediction

# Log the prediction
log_prediction(test_input, normalized_price, denormalized_price)
