"""
Physics-Guided Baseline Generator
=================================
Calculates optimal starting spaces using solid rocket motor physics equations.
This primes the optimizers toward valid spaces rather than naive random search.
"""
import numpy as np

def calculate_initial_bounds(constraints: dict, mode: str = "fast") -> list:
    """
    Derives sensible geometric search bounds using mass conservation and 
    common Kn (Klemmung) rules-of-thumb for a KNDX motor.
    """
    od = constraints.get("grain_od_m", 0.035)
    target_mass = constraints.get("target_mass_kg", 0.1)
    
    # KNDX rules of thumb
    density = 1879.0
    volume_needed = target_mass / density
    
    # Typical L/D is 3 to 10 for total length
    # Approx 4 grains
    n_grains = constraints.get("total_grains", 4)
    avg_vol_per_grain = volume_needed / n_grains
    
    # V = pi/4 * (OD^2 - core^2) * L
    # We want bounds that allow V to be reachable.
    
    if mode == "fast":
        return [
            (0.003, od * 0.55),      # throat
            (0.005, od * 1.5),       # exit (>= throat)
            (0.010, 0.150),          # bates length
            (0.004, od * 0.85),      # bates core diameter
        ]
    else:
        return [
            (0.003, od * 0.55),      # throat
            (0.005, od * 1.5),       # exit
            (0.010, 0.150),          # finocyl length
            (0.004, od * 0.85),      # finocyl core diameter
            (2, 8),                  # fin count (integer)
            (0.001, od * 0.30),      # fin width
            (0.002, od * 0.40),      # fin length
            (0.010, 0.150),          # bates length
            (0.004, od * 0.85),      # bates core diameter
        ]
