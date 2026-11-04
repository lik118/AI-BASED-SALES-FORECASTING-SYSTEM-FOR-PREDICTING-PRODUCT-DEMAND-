"""
Energy Consumption Data Generator
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class EnergyDataGenerator:
    """Generate realistic energy consumption data."""
    
    def __init__(self, days=365, frequency='hourly'):
        self.days = days
        self.frequency = frequency
        self.base_consumption = 50
        
    def generate_consumption_data(self):
        """Generate energy consumption time series."""
        start_date = datetime(2023, 1, 1)
        if self.frequency == 'hourly':
            periods = self.days * 24
            freq = 'h'
        else:
            periods = self.days
            freq = 'D'
        
        dates = pd.date_range(start=start_date, periods=periods, freq=freq)
        consumption = []
        
        for i, date in enumerate(dates):
            base = self.base_consumption
            day_of_year = date.dayofyear
            seasonal = 30 * np.sin(2 * np.pi * day_of_year / 365)
            hour = date.hour
            daily = 40 * np.sin(2 * np.pi * hour / 24)
            noise = np.random.normal(0, 5)
            total = base + seasonal + daily + noise
            consumption.append(max(total, 10))
        
        df = pd.DataFrame({
            'timestamp': dates,
            'consumption_kwh': consumption,
            'hour': [d.hour for d in dates],
            'day_of_week': [d.dayofweek for d in dates],
            'month': [d.month for d in dates]
        })
        return df
    
    def generate_weather_data(self, consumption_df):
        """Generate weather data."""
        weather_data = []
        for idx, row in consumption_df.iterrows():
            date = row['timestamp']
            month = date.month
            base_temp = 15 + 15 * np.sin(2 * np.pi * month / 12)
            temperature = base_temp + np.random.normal(0, 3)
            humidity = 60 - (temperature - 15) * 2 + np.random.normal(0, 5)
            humidity = np.clip(humidity, 20, 95)
            weather_data.append({
                'timestamp': date,
                'temperature_c': temperature,
                'humidity_percent': humidity
            })
        return pd.DataFrame(weather_data)
    
    def generate_occupancy_data(self, consumption_df):
        """Generate occupancy patterns."""
        occupancy_data = []
        for idx, row in consumption_df.iterrows():
            date = row['timestamp']
            hour = date.hour
            day_of_week = date.dayofweek
            
            if hour < 6:
                base_occupancy = 0.1
            elif hour < 9:
                base_occupancy = 0.7
            elif hour < 17:
                base_occupancy = 0.9
            elif hour < 21:
                base_occupancy = 0.5
            else:
                base_occupancy = 0.2
            
            if day_of_week >= 5:
                base_occupancy *= 0.5
            
            occupancy = np.clip(base_occupancy + np.random.normal(0, 0.05), 0, 1)
            occupancy_data.append({
                'timestamp': date,
                'occupancy_ratio': occupancy
            })
        return pd.DataFrame(occupancy_data)

def main():
    """Generate and save energy data."""
    print("Generating energy consumption data...")
    generator = EnergyDataGenerator(days=365, frequency='hourly')
    consumption_df = generator.generate_consumption_data()
    weather_df = generator.generate_weather_data(consumption_df)
    occupancy_df = generator.generate_occupancy_data(consumption_df)
    energy_data = consumption_df.merge(weather_df, on='timestamp')
    energy_data = energy_data.merge(occupancy_df, on='timestamp')
    energy_data.to_csv('data/energy_data.csv', index=False)
    print(f"✓ Generated {len(energy_data)} records")
    print(f"✓ Consumption range: {energy_data['consumption_kwh'].min():.2f} - {energy_data['consumption_kwh'].max():.2f} kWh")
    return energy_data

if __name__ == "__main__":
    main()

