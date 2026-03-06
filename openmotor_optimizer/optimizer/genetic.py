"""
Genetic Algorithm Optimizer
===========================
Iterative differential evolution algorithm optimizing for specific cost metrics.
"""
from scipy.optimize import differential_evolution
from typing import Dict, Any
import numpy as np

from ..generator.physics_guided import calculate_initial_bounds
from ..generator.random_generator import build_motor_dicts_from_vector
from ..propellants.openmotor_adapter import build_config, build_nozzle, build_propellant, assemble_motor
from ..simulation.openmotor_runner import run_simulation

class GeneticOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.constraints = config.get("constraints", {})
        self.mode = self.constraints.get("mode", "fast")
        self.bounds = calculate_initial_bounds(self.constraints, self.mode)
        
        # Differential Evolution requires knowing which variables are integers (like num_fins)
        num_vars = len(self.bounds)
        if self.mode == "fast":
            self.integrality = [False] * num_vars
        else:
            self.integrality = [False, False, False, False, True, False, False, False, False]
            
    def _objective(self, x: np.ndarray) -> float:
        nozzle_dict, grains_list = build_motor_dicts_from_vector(x, self.constraints)
        
        if grains_list is None:
            return 1e8 # Heavy penalty for geometric invalidity
            
        nozzle = build_nozzle(nozzle_dict["throat"], nozzle_dict["exit"])
        config = build_config(self.constraints)
        prop = build_propellant(self.config.get("propellant", {}))
        
        motor = assemble_motor(nozzle, config, prop, grains_list)
        success, metrics = run_simulation(motor)
        
        if not success:
            return 1e7 # Simulation exploded
            
        # Calculate cost function from metrics
        target_mass = self.constraints.get("target_mass_kg", 0.1)
        mass_err = abs(metrics["propellant_mass_kg"] - target_mass) / target_mass
        
        pres_limit = self.constraints.get("max_pressure_Pa", 700 * 6894.757)
        pres_pen = max(0, (metrics["peak_pressure_Pa"] - pres_limit) / pres_limit) * 50
        
        flux_limit = self.constraints.get("max_mass_flux", 500)
        flux_pen = max(0, (metrics["peak_mass_flux"] - flux_limit) / flux_limit) * 50
        
        impulse = metrics["total_impulse_Ns"]
        
        # We want to match mass, respect limits, and maximize impulse (negative penalty)
        cost = (mass_err * 100) + pres_pen + flux_pen - (impulse * 0.05)
        return cost

    def optimize(self) -> np.ndarray:
        print(f"Starting Genetic / DE Optimization...")
        result = differential_evolution(
            func=self._objective,
            bounds=self.bounds,
            strategy="best1bin",
            maxiter=self.config.get("optimizer", {}).get("max_iter", 50),
            popsize=self.config.get("optimizer", {}).get("pop_size", 15),
            tol=1e-6,
            integrality=self.integrality,
            disp=True,
            workers=1 # Using 1 intentionally to avoid Windows multiprocessing pickle issues inside classes
        )
        print(f"Genetic Search completed. Success: {result.success}")
        return result.x
