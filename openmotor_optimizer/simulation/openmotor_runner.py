"""
OpenMotor Runner
================
Handles the sequential execution of a single OpenMotor simulation, capturing physics metrics.
"""
import warnings
from typing import Dict, Any, Tuple

# Try to import SimAlertLevel for warning/error analysis
try:
    from motorlib.simResult import SimAlertLevel
except ImportError:
    SimAlertLevel = None

def run_simulation(motor: 'Motor') -> Tuple[bool, Dict[str, Any]]:
    """
    Executes a simulation on a given OpenMotor Motor instance.
    
    Returns:
        success (bool): Whether the simulation completed without strictly critical errors.
        metrics (dict): Extracted performance scalars.
    """
    if motor is None:
        return False, {"error": "Motor object is None"}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Simulation might print outputs or raise C++ errors if not careful
            sim = motor.runSimulation()
    except Exception as e:
        return False, {"error": f"Exception during simulation: {str(e)}"}

    if not getattr(sim, 'success', False):
        return False, {"error": "Simulation failed natively"}
        
    if SimAlertLevel is not None:
        errors = sim.getAlertsByLevel(SimAlertLevel.ERROR)
        if len(errors) > 0:
            return False, {"error": "Simulation alerts contained ERROR level flags"}

    # Extract metrics
    try:
        metrics = {
            "peak_pressure_Pa": sim.getMaxPressure(),
            "peak_mass_flux": sim.getPeakMassFlux(),
            "total_impulse_Ns": sim.getImpulse(),
            "average_thrust_N": sim.getAverageForce(),
            "burn_time_s": sim.channels["time"].getLast(),
            "peak_thrust_N": sim.getMaxForce() if hasattr(sim, 'getMaxForce') else sim.channels["force"].getMax()
        }
        
        # Calculate Propellant Mass initially
        density = motor.propellant.getProperty("density")
        mass = sum(g.getVolumeAtRegression(0) * density for g in motor.grains)
        metrics["propellant_mass_kg"] = mass
        
        # Calculate Specific Impulse Iteratively if possible
        if mass > 0:
            metrics["specific_impulse_s"] = metrics["total_impulse_Ns"] / (mass * 9.80665)
        else:
            metrics["specific_impulse_s"] = 0.0

        return True, metrics

    except Exception as e:
        return False, {"error": f"Metric extraction failed: {str(e)}"}
