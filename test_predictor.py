# test_predictor.py
from predictor import predict_price

# Valid test input matching model training features
test_input = {
    "car_links": 0.125,
    "city": "Bangalore",
    "new_car_detail_0_it": 0,
    "new_car_detail_1_ft": "Petrol",
    "new_car_detail_2_bt": "Hatchback",
    "new_car_detail_3_km": "23477",
    "new_car_detail_4_transmission": "Automatic",
    "new_car_detail_5_ownerNo": 1,
    "new_car_detail_6_owner": "1st Owner",
    "new_car_detail_7_oem": "Hyundai",
    "new_car_detail_8_model": "Hyndai Santro",
    "new_car_detail_9_modelYear": 2018,
    "new_car_detail_10_centralVariantId": 6670,
    "new_car_detail_11_variantName": "Sportz AMT BSIV",
    "new_car_detail_13_priceActual": 585000,
    "new_car_detail_14_priceSaving": 0,
    "new_car_detail_15_priceFixedText": 0,
    "car_age": 7,
    "new_car_overview_1_top_Year of Manufacture": 2018,
    "new_car_specs_1_top_Seats": 5
}

predicted_price = predict_price(test_input)
print(f"🪙 Estimated Price: ₹{int(predicted_price):,}")
