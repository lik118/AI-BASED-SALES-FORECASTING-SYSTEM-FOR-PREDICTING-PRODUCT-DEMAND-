"""
Building Energy Consumption Forecasting System
Main application module
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
from src.data_generator import EnergyDataGenerator
from src.forecasting_models import EnsembleForecaster, evaluate_forecast
import matplotlib.pyplot as plt

def run_forecasting_system():
    """Run the complete forecasting system."""
    print("\n" + "="*60)
    print("BUILDING ENERGY CONSUMPTION FORECASTING SYSTEM")
    print("="*60)
    
    # Generate data
    print("\n[1/4] Generating energy consumption data...")
    generator = EnergyDataGenerator(days=365)
    consumption_df = generator.generate_consumption_data()
    weather_df = generator.generate_weather_data(consumption_df)
    occupancy_df = generator.generate_occupancy_data(consumption_df)
    
    energy_data = consumption_df.merge(weather_df, on='timestamp')
    energy_data = energy_data.merge(occupancy_df, on='timestamp')
    energy_data.to_csv('data/energy_data.csv', index=False)
    print(f"✓ Generated {len(energy_data)} records")
    print(f"✓ Consumption: {energy_data['consumption_kwh'].min():.2f} - {energy_data['consumption_kwh'].max():.2f} kWh")
    
    # Prepare data for forecasting
    print("\n[2/4] Preparing data for forecasting...")
    consumption_data = energy_data['consumption_kwh'].values
    train_size = int(len(consumption_data) * 0.8)
    train_data = consumption_data[:train_size]
    test_data = consumption_data[train_size:]
    
    print(f"✓ Training set: {len(train_data)} samples")
    print(f"✓ Test set: {len(test_data)} samples")
    
    # Train forecasting model
    print("\n[3/4] Training ensemble forecasting model...")
    forecaster = EnsembleForecaster()
    forecaster.fit(train_data)
    print("✓ Model training completed")
    
    # Make predictions
    print("\n[4/4] Making predictions...")
    predictions = forecaster.predict(train_data, steps=len(test_data))
    
    # Evaluate performance
    metrics = evaluate_forecast(test_data, predictions)
    print(f"✓ Prediction accuracy metrics:")
    print(f"  - RMSE: {metrics['rmse']:.4f} kWh")
    print(f"  - MAE: {metrics['mae']:.4f} kWh")
    print(f"  - MAPE: {metrics['mape']:.2f}%")
    print(f"  - R²: {metrics['r2']:.4f}")
    
    # Generate visualizations
    print("\n[5/5] Generating visualizations...")
    create_visualizations(energy_data, test_data, predictions, metrics)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_records': len(energy_data),
        'training_samples': len(train_data),
        'test_samples': len(test_data),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'consumption_stats': {
            'min': float(energy_data['consumption_kwh'].min()),
            'max': float(energy_data['consumption_kwh'].max()),
            'mean': float(energy_data['consumption_kwh'].mean()),
            'std': float(energy_data['consumption_kwh'].std())
        }
    }
    
    with open('results/forecast_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to results/forecast_results.json")
    print("\n" + "="*60)
    print("FORECASTING SYSTEM EXECUTION COMPLETED")
    print("="*60 + "\n")
    
    return results

def create_visualizations(energy_data, test_data, predictions, metrics):
    """Create visualization charts."""
    
    # 1. Consumption Over Time
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Building Energy Consumption Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Time series
    ax = axes[0, 0]
    ax.plot(energy_data['timestamp'], energy_data['consumption_kwh'], linewidth=1.5, label='Actual')
    ax.set_title('Annual Energy Consumption Pattern')
    ax.set_xlabel('Date')
    ax.set_ylabel('Consumption (kWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Forecast vs Actual
    ax = axes[0, 1]
    ax.plot(range(len(test_data)), test_data, 'b-', label='Actual', linewidth=2)
    ax.plot(range(len(predictions)), predictions, 'r--', label='Predicted', linewidth=2)
    ax.set_title('Test Period: Actual vs Predicted')
    ax.set_xlabel('Hours')
    ax.set_ylabel('Consumption (kWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Hourly Pattern
    ax = axes[1, 0]
    hourly_avg = energy_data.groupby('hour')['consumption_kwh'].mean()
    ax.bar(hourly_avg.index, hourly_avg.values, color='steelblue', alpha=0.7)
    ax.set_title('Average Hourly Consumption Pattern')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Consumption (kWh)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Performance Metrics
    ax = axes[1, 1]
    metrics_names = ['RMSE', 'MAE', 'MAPE/10', 'R² Score']
    metrics_values = [
        metrics['rmse'],
        metrics['mae'],
        metrics['mape'] / 10,
        metrics['r2']
    ]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    bars = ax.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    ax.set_title('Model Performance Metrics')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/energy_forecast_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: energy_forecast_analysis.png")
    plt.close()
    
    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_data = energy_data[['consumption_kwh', 'temperature_c', 'humidity_percent', 'occupancy_ratio']].corr()
    
    im = ax.imshow(correlation_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation_data.columns)))
    ax.set_yticks(range(len(correlation_data.columns)))
    ax.set_xticklabels(['Consumption', 'Temperature', 'Humidity', 'Occupancy'], rotation=45)
    ax.set_yticklabels(['Consumption', 'Temperature', 'Humidity', 'Occupancy'])
    
    # Add correlation values
    for i in range(len(correlation_data)):
        for j in range(len(correlation_data)):
            text = ax.text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='Correlation Coefficient')
    plt.title('Feature Correlation Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/correlation_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: correlation_analysis.png")
    plt.close()

if __name__ == "__main__":
    results = run_forecasting_system()

