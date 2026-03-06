"""
Evaluation Performance Metrics
==============================
Defines the data structures for performance metrics used across the optimization engine.
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PerformanceMetrics:
    peak_thrust_N: float
    average_thrust_N: float
    burn_time_s: float
    total_impulse_Ns: float
    peak_pressure_Pa: float
    peak_mass_flux: float
    specific_impulse_s: float
    propellant_mass_kg: float
    
    # Costs or penalties (used by optimizers)
    objective_cost: float = float('inf')
    is_valid: bool = False

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'PerformanceMetrics':
        return PerformanceMetrics(
            peak_thrust_N=data.get("peak_thrust_N", 0.0),
            average_thrust_N=data.get("average_thrust_N", 0.0),
            burn_time_s=data.get("burn_time_s", 0.0),
            total_impulse_Ns=data.get("total_impulse_Ns", 0.0),
            peak_pressure_Pa=data.get("peak_pressure_Pa", 0.0),
            peak_mass_flux=data.get("peak_mass_flux", 0.0),
            specific_impulse_s=data.get("specific_impulse_s", 0.0),
            propellant_mass_kg=data.get("propellant_mass_kg", 0.0),
            is_valid=True
        )
