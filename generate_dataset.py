import pandas as pd
import numpy as np
import random

def generate_aqi_data(num_samples=5000):
    np.random.seed(42)
    random.seed(42)
    
    # Simulating values for standard air pollutants
    # PM2.5 (ug/m3), PM10 (ug/m3), Temperature (C), Humidity (%), CO (ppm), NO2 (ppb)
    
    pm25 = np.random.uniform(5, 250, num_samples)
    pm10 = pm25 * np.random.uniform(1.2, 2.5, num_samples) 
    temperature = np.random.uniform(15, 35, num_samples)
    humidity = np.random.uniform(30, 80, num_samples)
    co = np.random.uniform(0.1, 10, num_samples)
    no2 = np.random.uniform(10, 150, num_samples)
    
    # Calculate a synthetic AQI based on the parameters (Simplified calculation for mock dataset)
    # Higher pollutants = Higher AQI
    aqi_base = (pm25 * 0.8) + (pm10 * 0.4) + (co * 5) + (no2 * 0.2)
    
    # Add some non-linear relationships to make ANN useful
    aqi = aqi_base * (1 + (temperature - 25)*0.01) * (1 + (humidity - 50)*0.005)
    
    # Adding some noise
    aqi = aqi + np.random.normal(0, 10, num_samples)
    
    # Ensure AQI is positive and somewhat bounded
    aqi = np.clip(aqi, 10, 500)
    
    df = pd.DataFrame({
        'PM2.5': np.round(pm25, 2),
        'PM10': np.round(pm10, 2),
        'Temperature': np.round(temperature, 2),
        'Humidity': np.round(humidity, 2),
        'CO': np.round(co, 2),
        'NO2': np.round(no2, 2),
        'AQI': np.round(aqi, 0)
    })
    
    df.to_csv('air_quality_data.csv', index=False)
    print(f"Dataset generated with {num_samples} samples. Saved as 'air_quality_data.csv'.")

if __name__ == '__main__':
    generate_aqi_data()
