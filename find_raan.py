import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import esmatpy.core as esmat

kernel_dir = r"C:\Users\burak\OneDrive\Belgeler\AntgravityProjects\Yeni klasör\MySimulation3\datas"
esmat.load_kernel(f"{kernel_dir}\\de440.bsp")
esmat.load_kernel(f"{kernel_dir}\\naif0012.tls")
esmat.load_kernel(f"{kernel_dir}\\pck00011.tpc")
esmat.load_kernel(f"{kernel_dir}\\gm_de440.tpc")

sim_settings = esmat.SimulationSettings()
sim_settings.start_date = "2026-AUG-15 12:00:00"
sim_settings.end_date   = "2027-AUG-15 12:00:00"

nav_plan = esmat.NavigationPlan()
nav_plan.add_target(target_id=301, objective=esmat.MissionObjective.Flyby, target_alt_km=100.0)

best_raan = 0.0
best_inc = 0.0
best_dv = 999999.0

# Scan inclination and RAAN
for inc in range(10, 60, 5):
    for raan in range(0, 360, 10):
        sc = esmat.Spacecraft.from_keplerian(
            name="TestSat",
            center_id=399,
            altitude_km=200.0,
            eccentricity=0.000000,
            inclination=float(inc),
            raan=float(raan),
            arg_periapsis=0.0000,
            true_anomaly=0.0000,
            state_vector_epoch_utc="2026-AUG-15 12:00:00",
            mission_start_epoch_utc="2026-AUG-15 12:00:00"
        )
        try:
            burn_table = esmat.calculate_navigation_plan(
                spacecraft=sc, 
                simulation=sim_settings,
                targets=nav_plan,
                initial_delay_days=0.0
            )
            for burn in burn_table:
                if burn.trigger == esmat.TriggerType.GET:
                    dv_mag = math.sqrt(burn.dvx**2 + burn.dvy**2 + burn.dvz**2)
                    if dv_mag > 0.1 and dv_mag < best_dv:
                        best_dv = dv_mag
                        best_raan = raan
                        best_inc = inc
        except Exception as e:
            pass

print(f"Best Inc: {best_inc}, Best RAAN: {best_raan}, Minimum Delta-V: {best_dv:.2f} km/s")
