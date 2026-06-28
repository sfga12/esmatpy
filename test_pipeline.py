import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import esmatpy.core as esmat

# Kernels
kernel_dir = r"../datas"
esmat.load_kernel(f"{kernel_dir}/de440.bsp")
esmat.load_kernel(f"{kernel_dir}/naif0012.tls")
esmat.load_kernel(f"{kernel_dir}/pck00011.tpc")
esmat.load_kernel(f"{kernel_dir}/gm_de440.tpc")

print("Kerneller Yuklendi.")

sim_settings = esmat.SimulationSettings()
sim_settings.start_date = "2026-08-15T12:00:00"
sim_settings.end_date   = "2027-08-15T12:00:00"
sim_settings.active_bodies = [399, 5, 4, 199, 301, 8, 9, 6, 10, 7, 299]

# Spacecraft (Keplerian)
sc = esmat.Spacecraft.from_keplerian(
    name="TestSat",
    center_id=399,
    altitude_km=200.0,
    eccentricity=0.000000,
    inclination=30.0000,
    raan=300.0000,
    arg_periapsis=0.0000,
    true_anomaly=0.0000,
    state_vector_epoch_utc="2026-08-15T12:00:00",
    mission_start_epoch_utc="2026-08-15T12:00:00"
)

print(f"Spacecraft '{sc.name}' olusturuldu (Pos: {sc.initial_pos}, Vel: {sc.initial_vel})")

nav_plan = esmat.NavigationPlan()
nav_plan.add_target(target_id=301, objective=esmat.MissionObjective.OrbitInsertion, target_alt_km=300.0)

burn_table = esmat.calculate_navigation_plan(
    spacecraft=sc, 
    simulation=sim_settings,
    targets=nav_plan,
    initial_delay_days=0.0
)

import csv

print("\n--- Olusturulan Mission Table ---")
csv_data = [["Step", "Trigger", "Parameters", "dV_X (km/s)", "dV_Y (km/s)", "dV_Z (km/s)", "Total_dV (km/s)", "RefBody", "Frame"]]

for i, burn in enumerate(burn_table):
    total_dv = (burn.dvx**2 + burn.dvy**2 + burn.dvz**2)**0.5
    
    if burn.trigger == esmat.TriggerType.GET:
        trigger_name = "Time (GET)"
        params = f"{burn.get_h}:{burn.get_m}:{burn.get_s}"
    elif burn.trigger == esmat.TriggerType.APSIS:
        trigger_name = "Orbital Event"
        params = "Apoapsis" if burn.apsisType == 0 else "Periapsis"
    elif burn.trigger == esmat.TriggerType.ALTITUDE:
        trigger_name = "Altitude"
        ops = ["<", "<=", ">=", ">"]
        op_str = ops[burn.altCondition] if 0 <= burn.altCondition < 4 else "<="
        params = f"{op_str} {burn.targetAltKM:.1f} km"
        
    frame = 'VNB' if burn.isVNB else 'J2000'
    print(f"[{i+1}] Trigger: {trigger_name} | Params: {params} | dV: ({burn.dvx:.5f}, {burn.dvy:.5f}, {burn.dvz:.5f}) | Ref: {burn.refBodyID} | Frame: {frame}")
    csv_data.append([i+1, trigger_name, params, f"{burn.dvx:.5f}", f"{burn.dvy:.5f}", f"{burn.dvz:.5f}", f"{total_dv:.5f}", burn.refBodyID, frame])

csv_filename = "mission_plan.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_data)

print(f"\nTablo basariyla '{csv_filename}' dosyasina kaydedildi!")
