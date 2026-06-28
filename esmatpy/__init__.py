"""
ESMAT Python Data Library
Primary toolset for downloading, compiling, and analyzing solar wind metadata.
"""

from .enlil import fetch_available_runs, get_authoritative_timeline, get_enlil_data_intervals, load_enlil_dataset, create_cropped_enlil_dataset
from . import core

__all__ = [
    "fetch_available_runs",
    "get_authoritative_timeline",
    "get_enlil_data_intervals",
    "load_enlil_dataset",
    "create_cropped_enlil_dataset",
    "core",
    "print_burn_table"
]

import csv

def print_burn_table(burn_table, export_csv_path="mission_plan.csv"):
    """
    Prints the burn table in a formatted way and optionally exports it to a CSV file.
    """
    print(f"\n{'='*80}")
    print(f"{'ESMAT MISSION NAVIGATION PLAN':^80}")
    print(f"{'='*80}")
    
    csv_data = [["Step", "Trigger", "Parameters", "dV_X (km/s)", "dV_Y (km/s)", "dV_Z (km/s)", "Total_dV (km/s)", "RefBody", "Frame"]]

    for i, burn in enumerate(burn_table):
        total_dv = (burn.dvx**2 + burn.dvy**2 + burn.dvz**2)**0.5
        
        if burn.trigger == core.TriggerType.GET:
            trigger_name = "Time (GET)"
            params = f"{int(burn.get_h):02d}:{int(burn.get_m):02d}:{burn.get_s:05.2f}"
        elif burn.trigger == core.TriggerType.APSIS:
            trigger_name = "Orbital Event"
            params = "Apoapsis" if burn.apsisType == 0 else "Periapsis"
        elif burn.trigger == core.TriggerType.ALTITUDE:
            trigger_name = "Altitude"
            ops = ["<", "<=", ">=", ">"]
            op_str = ops[burn.altCondition] if 0 <= burn.altCondition < 4 else "<="
            
            body_map = {399: "EARTH", 301: "MOON", 199: "MERCURY", 299: "VENUS", 499: "MARS", 10: "SUN", 0: "CENT"}
            body_name = body_map.get(burn.altRefBodyID, str(burn.altRefBodyID))
            
            params = f"{body_name} {op_str} {burn.targetAltKM:.0f} km"
        else:
            trigger_name = "Unknown"
            params = ""
            
        frame = 'VNB' if burn.isVNB else 'J2000'
        ref_body = {399: "EARTH", 301: "MOON", 0: "CENT"}.get(burn.refBodyID, str(burn.refBodyID))
        
        if getattr(burn, "isDynamicCircularize", False):
            dv_str = f"dV: {'Dynamic Circularize':<29}"
            csv_data.append([i+1, trigger_name, params, "Dynamic", "Dynamic", "Dynamic", "Dynamic", ref_body, frame])
        else:
            dv_str = f"dV: ({burn.dvx:8.5f}, {burn.dvy:8.5f}, {burn.dvz:8.5f})"
            csv_data.append([i+1, trigger_name, params, f"{burn.dvx:.5f}", f"{burn.dvy:.5f}", f"{burn.dvz:.5f}", f"{total_dv:.5f}", ref_body, frame])
        
        print(f"[{i+1:02d}] Trigger: {trigger_name:<15} | Params: {params:<20} | {dv_str} | Ref: {ref_body:<5} | Frame: {frame}")

    print(f"{'-'*80}")
    
    if export_csv_path:
        with open(export_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_data)
        print(f"[!] Tablo basariyla '{export_csv_path}' dosyasina kaydedildi!")

