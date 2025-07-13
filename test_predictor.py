# test_predictor.py
from price_predictor import predict_price

# Constants for optional inverse scaling (if used originally)
MIN_PRICE = 100000
MAX_PRICE = 9600000

def inverse_price(predicted_value, min_price=100000, max_price=9600000):
    return predicted_value * (max_price - min_price) + min_price

# Valid test input matching model training features
test_input = {
    "car_links": 0.929425837320574,
    "city": "Bangalore",
    "new_car_detail_1_ft": "Petrol",
    "new_car_detail_2_bt": "Hatchback",
    "new_car_detail_3_km": "11949",
    "new_car_detail_4_transmission": "Manual",
    "new_car_detail_5_ownerNo": 1,
    "new_car_detail_6_owner": "1st Owner",
    "new_car_detail_7_oem": "Tata",
    "new_car_detail_8_model": "Tata Tiago",
    "new_car_detail_9_modelYear": 2018,
    "new_car_detail_10_centralVariantId": 2983,
    "new_car_detail_11_variantName": "1.2 Revotron XZ",
    "new_car_detail_13_priceActual": 0,
    "car_age": 7,
    "new_car_overview_1_top_Year of Manufacture": 2018
}

predicted_price = predict_price(test_input)
print(f"🪙 Estimated Price: ₹{int(predicted_price):,}")

