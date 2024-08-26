import firebase_admin
from firebase_admin import credentials, db
import csv
import os
from datetime import datetime, timedelta
from threading import Timer

import joblib
import numpy as np
# Load the Random Forest model, and the scaler.
loaded_model = joblib.load('/home/nextav/Desktop/marealb-iot/ML model/density-tds/random_forest_model.pkl')
loaded_scaler = joblib.load('/home/nextav/Desktop/marealb-iot/ML model/density-tds/scaler.pkl')

# Firebase credentials and initialization
cred = credentials.Certificate("./marealb-iot-firebase-adminsdk-6i4kf-6beb3b5e2f.json")  # The path in the pc
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://marealb-iot-default-rtdb.firebaseio.com'
})

# Paths to the CSV files
csv_file_path = "./sensor_data.csv"  # Main CSV file
today_csv_path = "./today_csv.csv"  # Today's CSV file

# Function to reset today's CSV file at midnight
def reset_today_csv():
    with open(today_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'date', 'system_temperature', 'tds', 'ext_temperature',
            'weather_temperature', 'weather_windDirection', 'weather_windSpeed',
            'weather_rainfall', 'ext_humidity', 'lidar_value', 'lidar_temperature', 'niveau_ngt', 'profondeur_eau','Sainity_from_tds'
        ])
    # Reschedule the reset for the next midnight
    schedule_midnight_reset()

# Schedule the reset of today's CSV at midnight
def schedule_midnight_reset():
    now = datetime.now()
    # Calculate the time remaining until midnight
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_until_midnight = (midnight - now).total_seconds()
    Timer(seconds_until_midnight, reset_today_csv).start()

# Function to handle new data from Firebase
def handle_new_data(event):
    data = event.data  # Assuming event.data contains the received data

    if data:
        try:
            if isinstance(data, dict):  # Check if data is already in dictionary format
                # Access data fields directly
                date = data.get('Date')
                if date is None:
                    raise ValueError("Date is None")
                
                # Try parsing the date with different formats
                try:
                    date_obj = datetime.strptime(date, "%Y-%m-%d")
                except ValueError:
                    try:
                        date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M")
                    except ValueError:
                        raise ValueError(f"Date format is incorrect: {date}")

                system_temperature = data.get('System temperature')
                tds = data.get('TDS ')
                ext_temperature = data.get('Exterior temperature')
                weather_temperature = data.get('Weather Temperature')
                weather_windDirection = data.get('Weather Wind direction')
                weather_windSpeed = data.get('Weather Wind speed')
                weather_rainfall = data.get('Weather Rainfall')
                ext_humidity = data.get('Exterior humidity')
                lidar_value = data.get('Lidar')
                lidar_temperature = data.get('Lidar temperature')
                nlidar_value = (-1) * float(lidar_value)
                niveau_ngt = (nlidar_value + 141) * 0.01
                profondeur_eau = (-1) * (1.6 + niveau_ngt)
                
                #tds2salinity
                loaded_model = joblib.load('/home/nextav/Desktop/marealb-iot/ML model/density-tds/random_forest_model.pkl')
                loaded_scaler = joblib.load('/home/nextav/Desktop/marealb-iot/ML model/density-tds/scaler.pkl')
                input_value = np.array([[tds,ext_temperature,weather_rainfall,ext_humidity,lidar_value]])  # Replace 'float_value' with your actual float value
                # Standardize the input value using the loaded scaler
                input_value_scaled = loaded_scaler.transform(input_value)
                # Predict using the loaded model
                Sainity_from_tds = round(loaded_model.predict(input_value_scaled)[0])
                

                # Append data to the main CSV file
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        date, system_temperature, tds, ext_temperature,
                        weather_temperature, weather_windDirection, weather_windSpeed,
                        weather_rainfall, ext_humidity, lidar_value, lidar_temperature
                    ])

                # Append data to today's CSV file if it's from today
                if date_obj.date() == datetime.now().date():
                    with open(today_csv_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            date, system_temperature, tds, ext_temperature,
                            weather_temperature, weather_windDirection, weather_windSpeed,
                            weather_rainfall, ext_humidity, lidar_value, lidar_temperature,niveau_ngt,profondeur_eau,Sainity_from_tds
                        ])
            else:
                # Handle other cases as needed
                pass
        except Exception as e:
            print("Error processing data:", e)
    else:
        print("Data received from Firebase")

# Function to retrieve existing data from Firebase
def retrieve_existing_data():
    ref = db.reference('/sensor_data')
    data = ref.get()

    if data:
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for key, entry in data.items():
                date = entry.get('Date')
                if date:
                    try:
                        # Try parsing the date with different formats
                        try:
                            date_obj = datetime.strptime(date, "%Y-%m-%d")
                        except ValueError:
                            date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M")
                        
                        writer.writerow([
                            date, entry.get('System temperature'), entry.get('TDS '), entry.get('Exterior temperature'),
                            entry.get('Weather Temperature'), entry.get('Weather Wind direction'), entry.get('Weather Wind speed'),
                            entry.get('Weather Rainfall'), entry.get('Exterior humidity'), entry.get('Lidar'), entry.get('Lidar temperature')
                        ])
                    except ValueError:
                        print(f"Date format is incorrect: {date}")
    else:
        print("No existing data found in Firebase")

# Check if main CSV file exists, if not create it with header
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'date', 'system_temperature', 'tds', 'ext_temperature',
            'weather_temperature', 'weather_windDirection', 'weather_windSpeed',
            'weather_rainfall', 'ext_humidity', 'lidar_value', 'lidar_temperature'
        ])

# Initialize today's CSV file and schedule its reset
reset_today_csv()
schedule_midnight_reset()

# Retrieve existing data from Firebase
retrieve_existing_data()

# Set up Firebase listener
ref = db.reference('/sensor_data')
ref.listen(handle_new_data)
