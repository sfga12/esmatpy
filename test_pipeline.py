import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import esmatpy.core as esmat

# Kernels
kernel_dir = r"datas" # Update this path to your local SPICE kernels folder
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

print("\n--- Olusturulan Mission Table ---")
for burn in burn_table:
    if burn.trigger == esmat.TriggerType.GET:
        print(f"Trigger: Time (GET) | Params: {burn.get_h}:{burn.get_m}:{burn.get_s} | dV: ({burn.dvx:.5f}, {burn.dvy:.5f}, {burn.dvz:.5f}) | Ref: {burn.refBodyID} | Frame: {'VNB' if burn.isVNB else 'J2000'}")
    
    elif burn.trigger == esmat.TriggerType.APSIS:
        apsis_name = "Apoapsis" if burn.apsisType == 0 else "Periapsis"
        print(f"Trigger: Orbital Event | Params: {apsis_name} | dV: ({burn.dvx:.5f}, {burn.dvy:.5f}, {burn.dvz:.5f}) | Ref: {burn.refBodyID} | Frame: {'VNB' if burn.isVNB else 'J2000'}")
