import os
import logging
from datetime import datetime

# Set up logging configuration
logging.basicConfig(
    filename='prediction_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define a function to log predictions
def log_prediction(input_data, estimated_price):
    log_entry = (
        f"INPUT: {input_data} | "
        f"ESTIMATED: {int(estimated_price):,}"
    )
    logging.info(log_entry)

