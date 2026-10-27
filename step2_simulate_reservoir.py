"""
Step 2: Reservoir mass balance simulation
Simulates Hirakud reservoir using real CWC observed inflows:
1. Standard Operating Policy (SOP)
2. Official CWC 1988 Rule Curve
Evaluates baseline reliability, flood routing, and drought deficits.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# 2. Hirakud Reservoir Physical Constants
FRL_ft = 630.0        # Full Reservoir Level (ft)
MDDL_ft = 590.0       # Minimum Drawdown / Dead Storage Level (ft)
S_DEAD = 840.0        # Dead storage capacity (MCM) at MDDL
S_GROSS = 5896.0      # Gross storage capacity (MCM) at FRL
S_LIVE = S_GROSS - S_DEAD # Live conservation storage: 5,056 MCM
AREA_FRL = 743.0      # Reservoir water spread area at FRL (sq km)

# Downstream safe discharge threshold to prevent delta flooding
SAFE_Q_CUMEC = 11326.0               # ~400,000 cfs
SAFE_Q_MCM = SAFE_Q_CUMEC * 0.0864   # ~978.6 MCM/day

# CWC 1988 Rule Curve Guide Elevations (Day of Year -> Permissible Level in ft)
# From July 1 (Day 182) to October 1 (Day 274)
CWC_DAYS = [182, 213, 223, 233, 244, 254, 264, 274]
CWC_ELEVS = [595.0, 606.0, 612.0, 617.0, 621.0, 626.0, 628.5, 630.0]

def storage_to_elevation(s_mcm):
    """Converts storage (MCM) to elevation (ft)."""
    s_clamped = np.clip(s_mcm, S_DEAD, S_GROSS)
    fraction = (s_clamped - S_DEAD) / S_LIVE
    return MDDL_ft + (FRL_ft - MDDL_ft) * (fraction ** (1.0 / 1.85))

def elevation_to_storage(elev_ft):
    """Converts elevation (ft) to storage (MCM)."""
    elev_clamped = np.clip(elev_ft, MDDL_ft, FRL_ft)
    fraction = (elev_clamped - MDDL_ft) / (FRL_ft - MDDL_ft)
    return S_DEAD + S_LIVE * (fraction ** 1.85)

def get_cwc_target_storage(doy):
    """Returns permissible storage (MCM) under CWC rule curve."""
    if doy < 182 or doy > 274:
        return S_GROSS # Full conservation outside monsoon
    target_elev = np.interp(doy, CWC_DAYS, CWC_ELEVS)
    return elevation_to_storage(target_elev)

# 3. Mass Balance Simulation Function
def simulate_reservoir(df, policy="SOP"):
    """
    Simulates daily storage continuity on real CWC data:
    S[t+1] = S[t] + Inflow[t] - Release_Irr[t] - Release_Flood[t] - Evap[t]
    """
    n = len(df)
    storage = np.zeros(n)
    elevation = np.zeros(n)
    rel_irr = np.zeros(n)
    rel_flood = np.zeros(n)
    deficit = np.zeros(n)
    rule_targets = np.zeros(n)

    current_s = S_GROSS

    for t in range(n):
        inflow = df.loc[t, "Inflow_MCM"]
        demand = df.loc[t, "Demand_MCM"]
        evap_rate = df.loc[t, "Evap_mm"]
        doy = df.loc[t, "DayOfYear"]

        # Evaporation loss
        area = AREA_FRL * ((current_s / S_GROSS) ** 0.68)
        evap_loss = area * evap_rate * 1e-3

        # Water available before release
        water_avail = current_s + inflow - evap_loss

        # 1. Irrigation Release
        avail_for_irr = max(0.0, water_avail - S_DEAD)
        r_irr = min(avail_for_irr, demand)
        rel_irr[t] = r_irr
        deficit[t] = max(0.0, demand - r_irr)
        water_left = water_avail - r_irr

        # 2. Rule Curve Target
        if policy == "CWC_RULE":
            s_target = get_cwc_target_storage(doy)
        else: # SOP
            s_target = S_GROSS
        rule_targets[t] = s_target

        # 3. Flood Release
        r_flood = 0.0
        if water_left > s_target:
            excess = water_left - s_target
            if policy == "SOP":
                r_flood = excess
                water_left = s_target
            else:
                evac = min(excess, SAFE_Q_MCM)
                if (water_left - evac) > S_GROSS:
                    evac += (water_left - evac - S_GROSS)
                r_flood = evac
                water_left -= r_flood

        rel_flood[t] = r_flood
        current_s = np.clip(water_left, S_DEAD, S_GROSS)
        storage[t] = current_s
        elevation[t] = storage_to_elevation(current_s)

    res_df = df.copy()
    res_df["Storage_MCM"] = storage
    res_df["Elevation_ft"] = elevation
    res_df["Release_Irr_MCM"] = rel_irr
    res_df["Release_Flood_MCM"] = rel_flood
    res_df["Total_Outflow_MCM"] = rel_irr + rel_flood
    res_df["Total_Outflow_cumec"] = res_df["Total_Outflow_MCM"] / 0.0864
    res_df["Deficit_MCM"] = deficit
    res_df["Rule_Target_MCM"] = rule_targets
    return res_df

# 4. Run simulations for both policies
print("Loading real CWC dataset...")
df_data = pd.read_csv(os.path.join(DATA_DIR, "hirakud_daily_data.csv"))
df_data["Date"] = pd.to_datetime(df_data["Date"])

print("Simulating Standard Operating Policy (SOP) on Real Data...")
sim_sop = simulate_reservoir(df_data, policy="SOP")

print("Simulating Official CWC 1988 Rule Curve on Real Data...")
sim_cwc = simulate_reservoir(df_data, policy="CWC_RULE")

# 5. Compute summary metrics on real data
def get_metrics(sim_df, name):
    tot_dem = sim_df["Demand_MCM"].sum()
    tot_irr = sim_df["Release_Irr_MCM"].sum()
    vol_reliability = (tot_irr / tot_dem) * 100.0
    peak_outflow = sim_df["Total_Outflow_cumec"].max()
    
    # Drought year deficit (e.g. 2002 or 2017)
    drought_def_2002 = sim_df[sim_df["Year"] == 2002]["Deficit_MCM"].sum()
    drought_def_2017 = sim_df[sim_df["Year"] == 2017]["Deficit_MCM"].sum()
    
    return {
        "Policy": name,
        "Volumetric_Reliability_%": round(vol_reliability, 1),
        "Peak_Flood_Outflow_cumec": round(peak_outflow, 1),
        "2002_Drought_Deficit_MCM": round(drought_def_2002, 1),
        "2017_Drought_Deficit_MCM": round(drought_def_2017, 1)
    }

metrics = [get_metrics(sim_sop, "Standard Operating Policy (SOP)"),
           get_metrics(sim_cwc, "Official CWC 1988 Rule Curve")]

summary_df = pd.DataFrame(metrics)
print("\n--- Real Data Baseline Performance Summary (2001-2025) ---")
print(summary_df.to_string(index=False))

# Save summary table
summary_csv = os.path.join(RESULTS_DIR, "table_baseline_summary.csv")
summary_df.to_csv(summary_csv, index=False)
print("\nSaved real summary table:", summary_csv)

# 6. Plot Baseline Results for Sample Flood Season (Monsoon 2003)
# 2003 was the record flood season (Aug 30 peak)
mask_sample = (df_data["Date"] >= "2003-07-01") & (df_data["Date"] <= "2003-10-31")
d_sub = df_data.loc[mask_sample, "Date"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

# (a) Outflow hydrographs
inflow_sample = df_data.loc[mask_sample, "Inflow_cumec"]
sop_outflow = sim_sop.loc[mask_sample, "Total_Outflow_cumec"]
cwc_outflow = sim_cwc.loc[mask_sample, "Total_Outflow_cumec"]

ax1.plot(d_sub, inflow_sample, "k-", linewidth=1.5, label=f"Real CWC Inflow (Peak: {inflow_sample.max():,.0f} m³/s)")
ax1.plot(d_sub, sop_outflow, "r--", linewidth=1.2, label="SOP Outflow")
ax1.plot(d_sub, cwc_outflow, "b-.", linewidth=1.3, label="CWC Rule Outflow")
ax1.axhline(SAFE_Q_CUMEC, color="crimson", linestyle=":", label="Safe Downstream Limit (11,326 m³/s)")
ax1.set_ylabel("Discharge (m³/s)")
ax1.set_title("(a) Real CWC Inflow and Simulated Outflow Discharges (Monsoon 2003)")
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=8.5)

# (b) Storage elevation vs Rule Curve
ax2.plot(d_sub, sim_sop.loc[mask_sample, "Elevation_ft"], "r--", linewidth=1.2, label="SOP Storage Level")
ax2.plot(d_sub, sim_cwc.loc[mask_sample, "Elevation_ft"], "b-.", linewidth=1.4, label="CWC Simulated Level")
cwc_el = [storage_to_elevation(s) for s in sim_cwc.loc[mask_sample, "Rule_Target_MCM"]]
ax2.plot(d_sub, cwc_el, "k:", linewidth=1.2, label="Official CWC Guide Level")
ax2.axhline(630.0, color="gray", linestyle="-", linewidth=1, label="FRL (630 ft)")
ax2.set_ylabel("Elevation (ft)")
ax2.set_xlabel("Date")
ax2.set_title("(b) Reservoir Storage Elevation Dynamics vs Rule Curve")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
ax2.legend(loc="lower right", fontsize=8.5)

plt.tight_layout()
fig_out = os.path.join(RESULTS_DIR, "fig2_baseline_simulation.png")
plt.savefig(fig_out, dpi=200)
plt.close()
print("Saved baseline simulation figure on real data:", fig_out)
