import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
 
np.random.seed(42)

def generate_hybrid_park_dataset():
    """
    Generate a 1-year synthetic hourly time-series dataset for a Hybrid Power Park
    with Solar and Coal capacity in a high-heat desert environment (e.g., Rajasthan).

    Returns:
        DataFrame with 8,760 hourly records
    """

    start_date = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(hours=i) for i in range(8760)]

    data = {
        'Timestamp': timestamps,
        'Hour_of_Day': [ts.hour for ts in timestamps],
        'Temperature_C': [],
        'Cloud_Cover_Pct': [],
        'Demand_MW': [],
        'Solar_MW': []
    }

    for i, ts in enumerate(timestamps):
        day_of_year = ts.timetuple().tm_yday
        hour = ts.hour
        month = ts.month

        seasonal_avg = 25.5 + 17.5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

        daily_offset = 12 * np.sin(2 * np.pi * (hour - 4) / 24)
        noise = np.random.normal(0, 0.5)

        temp = seasonal_avg + daily_offset + noise
        data['Temperature_C'].append(max(temp, 0))


        base_cloud = np.random.uniform(0.0, 0.15)  # Clear skies baseline

        # Monsoon season (July-August): sustained high cloud cover
        if month in [7, 8]:
            # 60% chance of monsoon clouds in Jul-Aug
            monsoon_cloud = np.random.uniform(0.4, 0.85)
            cloud = monsoon_cloud if np.random.random() < 0.6 else base_cloud
        else:
            # 5% chance of random cloudy day spikes in other months
            cloud = np.random.uniform(0.5, 0.9) if np.random.random() < 0.05 else base_cloud

        data['Cloud_Cover_Pct'].append(np.clip(cloud, 0.0, 1.0))

        # ============ DEMAND (d2 - Electricity Demand) ============
        # Base load: 1000 MW
        demand = 1000.0

        # Morning peak (08:00 - 10:00): smooth rise and fall
        if 8 <= hour <= 10:
            demand += 150 * np.sin(np.pi * (hour - 8) / 3)

        # Evening peak (18:00 - 22:00): massive spike (THE DUCK CURVE!)
        if 18 <= hour <= 22:
            demand += 400 * np.sin(np.pi * (hour - 18) / 4)

        # Temperature multiplier for AC load (hot days = higher demand)
        temp_value = data['Temperature_C'][-1]
        if temp_value > 30:
            # Each degree above 30°C adds ~1% to demand (max 1.15x at 45°C)
            ac_multiplier = 1 + 0.15 * (temp_value - 30) / 15
            demand *= ac_multiplier

        # Random Gaussian noise (±2% standard deviation)
        noise_factor = np.random.normal(1.0, 0.02)
        demand *= noise_factor

        data['Demand_MW'].append(max(demand, 500))  # Floor at 500 MW

        # ============ SOLAR (d1 - Solar Production) ============
        # Zero from 19:00 to 05:59
        if hour < 6 or hour >= 19:
            solar = 0.0
        else:
            # Bell curve from 06:00 to 18:00, peak of 800 MW at 12:00 (noon)
            # Using Gaussian distribution with mean=12, sigma=3.5
            mean_hour = 12.0
            std_dev = 3.5
            normalized_hour = (hour - mean_hour) / std_dev

            # Bell curve peaks at exactly 800 MW at 12:00
            bell_curve = 800 * np.exp(-0.5 * normalized_hour ** 2)

            # Cloud cover multiplier: reduces solar output linearly
            # If cloud_cover = 0.8 (80%), then solar = 20% of theoretical
            cloud_factor = 1 - data['Cloud_Cover_Pct'][-1]
            solar = bell_curve * cloud_factor

        data['Solar_MW'].append(max(solar, 0))

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    csv_path = 'hybrid_park_dataset.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Dataset saved to {csv_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

    return df


def visualize_duck_curve(date_str='2025-07-15'):
    """
    Plot a single hot summer day to visualize the Duck Curve clash.
    Creates a 2x2 subplot showing Solar+Demand, Net Load, Temperature, and Cloud Cover.

    Args:
        date_str: Date to visualize (default: hot day in mid-July monsoon)
    """
    df = pd.read_csv('hybrid_park_dataset.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Filter for the requested date
    date_obj = pd.to_datetime(date_str)
    day_data = df[df['Timestamp'].dt.date == date_obj.date()].copy()

    if len(day_data) == 0:
        print(f"No data for {date_str}")
        return

    day_data = day_data.sort_values('Hour_of_Day')
    day_data['Hour'] = day_data['Hour_of_Day']

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Hybrid Power Park - {date_str}\nDuck Curve Analysis',
                 fontsize=16, fontweight='bold')

    # --- Plot 1: Solar vs Demand (overlaid) ---
    ax1 = axes[0, 0]
    ax1.fill_between(day_data['Hour'], 0, day_data['Solar_MW'],
                      alpha=0.4, color='orange', label='Solar (Available)')
    ax1.plot(day_data['Hour'], day_data['Solar_MW'], marker='o', linewidth=2.5,
             label='Solar Production', color='#FF8C00', markersize=5)
    ax1.plot(day_data['Hour'], day_data['Demand_MW'], marker='s', linewidth=2.5,
             label='Electricity Demand', color='#0047AB', markersize=5)
    ax1.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Power (MW)', fontsize=11, fontweight='bold')
    ax1.set_title('Solar Production vs Electricity Demand', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_xlim(-0.5, 23.5)

    # --- Plot 2: Net Load (The DUCK CURVE!) ---
    net_load = day_data['Demand_MW'] - day_data['Solar_MW']
    net_load_max = net_load.max()

    colors = []
    for x in net_load:
        if x > 1200:
            colors.append('#8B0000')  # Dark red - critical
        elif x > 1100:
            colors.append('#FF4500')  # Orange-red - high
        elif x < 900:
            colors.append('#228B22')  # Forest green - low
        else:
            colors.append('#FFD700')  # Gold - normal

    ax2 = axes[0, 1]
    ax2.bar(day_data['Hour'], net_load, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1000, color='black', linestyle='--', linewidth=1.5, label='Base Load (1000 MW)')
    ax2.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Net Load (MW)', fontsize=11, fontweight='bold')
    ax2.set_title('THE DUCK CURVE: Net Load (Demand - Solar)', fontsize=12, fontweight='bold', color='red')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlim(-0.5, 23.5)

    # --- Plot 3: Temperature ---
    ax3 = axes[1, 0]
    ax3.fill_between(day_data['Hour'], day_data['Temperature_C'],
                     alpha=0.5, color='#FF6347')  # Tomato
    ax3.plot(day_data['Hour'], day_data['Temperature_C'], marker='D',
             color='#8B0000', linewidth=2.5, markersize=5)
    ax3.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
    ax3.set_title('Ambient Temperature (Drives AC Load)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xticks(range(0, 24, 2))
    ax3.set_xlim(-0.5, 23.5)

    # --- Plot 4: Cloud Cover ---
    ax4 = axes[1, 1]
    ax4.fill_between(day_data['Hour'], day_data['Cloud_Cover_Pct'] * 100,
                     alpha=0.5, color='#708090')  # Slate gray
    ax4.plot(day_data['Hour'], day_data['Cloud_Cover_Pct'] * 100, marker='^',
             color='#2F4F4F', linewidth=2.5, markersize=5)
    ax4.set_xlabel('Hour of Day', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Cloud Cover (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Cloud Cover (Reduces Solar Output)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xticks(range(0, 24, 2))
    ax4.set_xlim(-0.5, 23.5)
    ax4.set_ylim([0, 100])

    plt.tight_layout()

    # Save figure
    fig_path = 'duck_curve_visualization.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved to {fig_path}")

    plt.show()

    # Print detailed day summary
    print(f"\n{'='*70}")
    print(f"HYBRID POWER PARK ANALYSIS: {date_str}")
    print(f"{'='*70}")
    print(f"Temperature Range:  {day_data['Temperature_C'].min():.1f}°C - {day_data['Temperature_C'].max():.1f}°C")
    print(f"Cloud Cover Range:  {day_data['Cloud_Cover_Pct'].min()*100:.1f}% - {day_data['Cloud_Cover_Pct'].max()*100:.1f}%")
    print(f"Demand Range:       {day_data['Demand_MW'].min():.1f} MW - {day_data['Demand_MW'].max():.1f} MW")
    print(f"Solar Range:        {day_data['Solar_MW'].min():.1f} MW - {day_data['Solar_MW'].max():.1f} MW")
    print(f"\nPeak Solar:         {day_data['Solar_MW'].max():.1f} MW @ {int(day_data.loc[day_data['Solar_MW'].idxmax(), 'Hour_of_Day']):02d}:00")
    print(f"Peak Demand:        {day_data['Demand_MW'].max():.1f} MW @ {int(day_data.loc[day_data['Demand_MW'].idxmax(), 'Hour_of_Day']):02d}:00")
    peak_net = net_load.max()
    peak_net_hour = day_data.loc[net_load.idxmax(), 'Hour_of_Day']
    print(f"Peak Net Load:      {peak_net:.1f} MW @ {int(peak_net_hour):02d}:00 ⚠️  (DUCK CURVE CLASH)")
    print(f"Coal Ramp-Down:     Solar rises {day_data['Solar_MW'].max():.1f} MW in 6 hours (06:00-12:00)")
    print(f"Coal Ramp-Up:       Solar falls {day_data['Solar_MW'].max():.1f} MW in 6 hours (12:00-18:00)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print("="*70)
    print("HYBRID POWER PARK - SYNTHETIC DATA GENERATOR")
    print("Location: High-Heat Desert (Rajasthan-like conditions)")
    print("="*70 + "\n")

    # Generate the dataset
    print("▶ Generating 1-year synthetic dataset (8,760 hourly records)...\n")
    df = generate_hybrid_park_dataset()

    # Show sample
    print("\nSample of generated data (first 10 rows):")
    print(df.head(10).to_string(index=False))

    # Visualize the duck curve for a sample date
    test_date = '2025-07-15'
    print("\n" + "="*70)
    print("VISUALIZATION: Duck Curve Analysis")
    print("="*70)
    visualize_duck_curve(test_date)

    print("\n" + "="*70)
    print("✓ All files generated successfully!")
    print("  - hybrid_park_dataset.csv (8,760 rows)")
    print("  - duck_curve_visualization.png (4-panel analysis)")
    print("="*70)
