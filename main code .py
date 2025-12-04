# Name: satyam bhardwaj
# Roll number: 2501730293
# Course Code: ETCCPP102
# Capstone Assignment: Campus Energy-Use Dashboard
# Date: 2025-12-08

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import logging

# --- Configuration ---
DATA_DIR = Path('.') # Assumes CSVs are in the root directory for simplicity in this environment
OUTPUT_DIR = Path('output')
LOG_FILE = 'energy_dashboard.log'
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Task 3: Object-Oriented Modeling ---

class MeterReading:
    """Represents a single energy consumption reading."""
    def __init__(self, timestamp, kwh):
        self.timestamp = pd.to_datetime(timestamp)
        self.kwh = float(kwh)
        
    def __repr__(self):
        return f"MeterReading({self.timestamp.strftime('%Y-%m-%d %H:%M')}, {self.kwh:.2f} kWh)"

class Building:
    """Represents a single building with energy consumption data."""
    def __init__(self, name):
        self.name = name
        self.meter_readings = [] # List of MeterReading objects

    def add_reading(self, timestamp, kwh):
        """Adds a new MeterReading to the building."""
        try:
            reading = MeterReading(timestamp, kwh)
            self.meter_readings.append(reading)
        except ValueError as e:
            logging.warning(f"Skipping invalid reading for {self.name}: {e}")

    def get_dataframe(self):
        """Converts readings into a Pandas DataFrame for analysis."""
        if not self.meter_readings:
            return pd.DataFrame()
        data = [{'Timestamp': r.timestamp, 'Usage_kwh': r.kwh} for r in self.meter_readings]
        df = pd.DataFrame(data).set_index('Timestamp')
        return df
        
    def calculate_total_consumption(self):
        """Calculates the total consumption from all readings."""
        return sum(r.kwh for r in self.meter_readings)

class BuildingManager:
    """Manages all Building objects and performs campus-wide analysis."""
    def __init__(self):
        self.buildings = {}

    def add_building(self, name, df_readings):
        """Creates a Building object and populates it with DataFrame data."""
        building = Building(name)
        for index, row in df_readings.iterrows():
            building.add_reading(index, row['Usage_kwh'])
        self.buildings[name] = building
        logging.info(f"Building '{name}' added with {len(building.meter_readings)} readings.")
        return building

# --- Task 1: Data Ingestion and Validation ---

def ingest_data(data_dir):
    """Loops through the directory, reads multiple CSVs, and merges them."""
    all_data = []
    
    # Identify CSV files using glob for simplicity in this environment
    csv_files = [f for f in data_dir.glob('data_building_*.csv')]
    
    if not csv_files:
        print(f"Error: No CSV files found in {data_dir}. Exiting.")
        return pd.DataFrame()

    for file_path in csv_files:
        building_name = file_path.stem.replace('data_building_', '').upper()
        print(f"Processing data for: {building_name}")
        try:
            # Task 1: Handle corrupt data with on_bad_lines='skip'
            df = pd.read_csv(file_path, on_bad_lines='skip')
            
            # Basic validation
            if 'Timestamp' not in df.columns or 'Usage_kwh' not in df.columns:
                logging.error(f"Skipping {file_path.name}: Missing required columns.")
                continue

            # Task 1: Add metadata
            df['Building'] = building_name
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce') # Errors='coerce' for handling bad dates
            df = df.dropna(subset=['Timestamp', 'Usage_kwh'])

            all_data.append(df)
            logging.info(f"Successfully ingested data for {building_name}.")
            
        except FileNotFoundError:
            logging.error(f"File not found: {file_path.name}")
        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")

    if not all_data:
        return pd.DataFrame()
        
    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined = df_combined.set_index('Timestamp').sort_index()
    print(f"Total records in combined DataFrame: {len(df_combined)}")
    return df_combined

# --- Task 2: Core Aggregation Logic ---

def calculate_daily_totals(df):
    """Calculates total campus consumption per day."""
    if df.empty: return pd.DataFrame()
    # Use .resample('D') for daily totals
    return df['Usage_kwh'].resample('D').sum().rename('Daily_Total_kwh')

def calculate_weekly_aggregates(df):
    """Calculates total campus consumption per week."""
    if df.empty: return pd.DataFrame()
    # Use .resample('W') for weekly aggregates
    return df['Usage_kwh'].resample('W').agg(['sum', 'mean', 'max'])

def building_wise_summary(df):
    """Creates a summary table per building (mean, min, max, total)."""
    if df.empty: return {}
    summary = df.groupby('Building')['Usage_kwh'].agg(['mean', 'min', 'max', 'sum']).rename(
        columns={'sum': 'Total_kwh'}
    ).sort_values(by='Total_kwh', ascending=False)
    return summary

# --- Task 4: Visual Output with Matplotlib ---

def generate_dashboard_plots(df_combined):
    """Generates a unified Matplotlib figure with all required charts."""
    if df_combined.empty: 
        print("Cannot generate plots: No combined data.")
        return

    # Task 4: Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    plt.suptitle('Campus Energy-Use Dashboard', fontsize=18)
    
    # Chart 1 (Top Left): Trend Line – daily consumption over time for all buildings
    daily_df = df_combined.groupby('Building')['Usage_kwh'].resample('D').sum().unstack(level=0).fillna(0)
    daily_df.plot(ax=axes[0, 0], kind='line', marker='o')
    axes[0, 0].set_title('Daily Energy Consumption Trend by Building')
    axes[0, 0].set_ylabel('Total kWh')
    axes[0, 0].grid(True, linestyle='--')
    
    # Chart 2 (Top Right): Bar Chart – compare average weekly usage across buildings
    # Aggregate to weekly average consumption per building
    weekly_avg = df_combined.groupby(['Building', pd.Grouper(freq='W')])['Usage_kwh'].sum().groupby('Building').mean()
    weekly_avg.plot(ax=axes[0, 1], kind='bar', rot=0)
    axes[0, 1].set_title('Average Weekly Total Consumption per Building')
    axes[0, 1].set_ylabel('Avg Weekly kWh')
    axes[0, 1].grid(axis='y', linestyle='--')
    
    # Chart 3 (Bottom Left): Scatter Plot – plot peak-hour consumption vs. building
    # Calculate consumption grouped by hour-of-day and building (Peak Load Time)
    peak_hour_df = df_combined.reset_index()
    peak_hour_df['Hour'] = peak_hour_df['Timestamp'].dt.hour
    hourly_mean = peak_hour_df.groupby(['Building', 'Hour'])['Usage_kwh'].mean().reset_index()
    
    for name, group in hourly_mean.groupby('Building'):
        axes[1, 0].scatter(group['Hour'], group['Usage_kwh'], label=name, alpha=0.7)
    
    axes[1, 0].set_title('Hourly Mean Consumption (Peak Load Analysis)')
    axes[1, 0].set_xlabel('Hour of Day (24h)')
    axes[1, 0].set_ylabel('Mean Consumption (kWh)')
    axes[1, 0].legend(title='Building')
    axes[1, 0].grid(True, linestyle='--')

    # Chart 4 (Bottom Right): Placeholder / Additional Chart (Daily vs Total)
    daily_totals = calculate_daily_totals(df_combined)
    daily_totals.plot(ax=axes[1, 1], kind='area', alpha=0.5)
    axes[1, 1].set_title('Campus-Wide Daily Total Consumption')
    axes[1, 1].set_ylabel('Total Daily kWh')
    axes[1, 1].grid(True, linestyle='--')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    
    # Save the chart
    dashboard_path = OUTPUT_DIR / 'dashboard.png'
    plt.savefig(dashboard_path)
    plt.close()
    print(f"\n- Dashboard saved to {dashboard_path}")
    
# --- Task 5: Persistence and Executive Summary ---

def export_and_summarize(df_combined, building_summary):
    """Exports data and creates a text summary."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Export: cleaned_energy_data.csv
    df_combined.to_csv(OUTPUT_DIR / 'cleaned_energy_data.csv')
    print(f"- Cleaned data exported to {OUTPUT_DIR / 'cleaned_energy_data.csv'}")

    # Export: building_summary.csv
    building_summary.to_csv(OUTPUT_DIR / 'building_summary.csv')
    print(f"- Building summary exported to {OUTPUT_DIR / 'building_summary.csv'}")

    # Create Executive Summary (summary.txt)
    total_campus_consumption = building_summary['Total_kwh'].sum()
    highest_consumer = building_summary.index[0]
    peak_hour_data = df_combined.reset_index()
    peak_hour_data['Hour'] = peak_hour_data['Timestamp'].dt.hour
    
    # Find the hour with the highest overall mean consumption
    peak_hour_kwh = peak_hour_data.groupby('Hour')['Usage_kwh'].mean().idxmax()

    summary_content = f"""
*** EXECUTIVE ENERGY SUMMARY ***
Date Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Total Campus Consumption (Observed Period): {total_campus_consumption:.2f} kWh
Analysis Period: {df_combined.index.min().strftime('%Y-%m-%d')} to {df_combined.index.max().strftime('%Y-%m-%d')}

1. HIGHEST CONSUMER:
   - Building: {highest_consumer}
   - Total Consumption: {building_summary.loc[highest_consumer, 'Total_kwh']:.2f} kWh

2. PEAK LOAD TIME:
   - The campus-wide average consumption peaks consistently around Hour {peak_hour_kwh}:00.
   - Recommendation: Investigate activities contributing to high load during this hour.

3. TRENDS:
   - Weekly usage comparisons (Bar Chart) clearly show Building {highest_consumer} has the largest average weekly footprint.
   - The daily trend line shows energy usage is stable but could have significant spikes on certain days, which need further investigation.

(End of Summary)
"""
    summary_path = OUTPUT_DIR / 'executive_summary.txt'
    with summary_path.open('w') as f:
        f.write(summary_content.strip())
        
    print(f"- Executive summary saved to {summary_path}")
    print("\n--- EXECUTIVE SUMMARY (CONSOLE) ---")
    print(summary_content.strip())
    
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Starting Capstone Project: Campus Energy Dashboard")
    
    # 1. Data Ingestion
    df_combined = ingest_data(DATA_DIR)
    if df_combined.empty:
        return

    # 3. OOP Modeling (Convert data back to OOP structure for reporting)
    manager = BuildingManager()
    for name in df_combined['Building'].unique():
        df_building = df_combined[df_combined['Building'] == name].drop(columns=['Building'])
        manager.add_building(name, df_building)
        
    # 2. Core Aggregation Logic
    building_summary_df = building_wise_summary(df_combined)

    # 4. Visualization
    generate_dashboard_plots(df_combined.drop(columns=['Building']))

    # 5. Persistence and Executive Summary
    export_and_summarize(df_combined, building_summary_df)

if __name__ == "__main__":
    main()
