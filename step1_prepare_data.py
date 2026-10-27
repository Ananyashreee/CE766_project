"""
Step 1: Prepare real hydrological data
Loads the authentic daily streamflow dataset downloaded from the
National Water Data Portal (NWIC / Ministry of Jal Shakti),
calculates key hydrological statistics, and plots the real inflow patterns.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 2. Load Real Government Data
csv_file = os.path.join(DATA_DIR, "hirakud_daily_data.csv")
df = pd.read_csv(csv_file)
df["Date"] = pd.to_datetime(df["Date"])

print("=" * 70)
print("REAL HYDROLOGICAL DATA SUMMARY (Central Water Commission / NWIC)")
print("Station: Basantpur (Mahanadi River at Hirakud Entry, 2001–2025)")
print("=" * 70)

# 3. Calculate statistics from real data
annual_bcm = df.groupby("Year")["Inflow_MCM"].sum() / 1000.0 # in BCM
monsoon_flow = df[df["Month"].isin([6, 7, 8, 9, 10])]["Inflow_MCM"].sum()
total_flow = df["Inflow_MCM"].sum()
monsoon_pct = (monsoon_flow / total_flow) * 100.0

peak_inflow = df["Inflow_cumec"].max()
peak_date = df.loc[df["Inflow_cumec"].idxmax(), "Date"].date()

print(f"Total days analyzed: {len(df)}")
print(f"Period: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Mean annual inflow: {annual_bcm.mean():.2f} BCM/year")
print(f"Wettest year ({annual_bcm.idxmax()}): {annual_bcm.max():.2f} BCM")
print(f"Driest year ({annual_bcm.idxmin()}): {annual_bcm.min():.2f} BCM")
print(f"Peak recorded flood inflow: {peak_inflow:,.1f} m³/s (Recorded on {peak_date})")
print(f"Monsoon concentration (June–October): {monsoon_pct:.1f}%")

# 4. Plot Real Inflow Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# (a) Annual Inflow Volume (BCM)
ax1.bar(annual_bcm.index, annual_bcm.values, color="steelblue", edgecolor="black", alpha=0.85)
ax1.axhline(annual_bcm.mean(), color="crimson", linestyle="--", linewidth=1.5, label=f"Mean: {annual_bcm.mean():.1f} BCM")
ax1.set_xlabel("Year")
ax1.set_ylabel("Annual Inflow (BCM)")
ax1.set_title("(a) Real Annual Inflow to Hirakud (2001–2025)")
ax1.legend()
ax1.grid(True, linestyle=":", alpha=0.6)

# (b) Monthly Inflow Volume (BCM)
monthly_bcm = df.groupby("Month")["Inflow_MCM"].sum() / (len(annual_bcm) * 1000.0)
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ax2.bar(range(1, 13), monthly_bcm, color="seagreen", edgecolor="black", alpha=0.85)
ax2.set_xticks(range(1, 13))
ax2.set_xticklabels(month_names, rotation=30, ha="right")
ax2.set_xlabel("Month")
ax2.set_ylabel("Average Monthly Inflow (BCM)")
ax2.set_title("(b) Real Monthly Inflow Distribution (2001–2025)")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
fig_path = os.path.join(RESULTS_DIR, "fig1_inflow_summary.png")
plt.savefig(fig_path, dpi=200)
plt.close()
print(f"\nSaved real inflow summary figure: {fig_path}")
print("=" * 70)
