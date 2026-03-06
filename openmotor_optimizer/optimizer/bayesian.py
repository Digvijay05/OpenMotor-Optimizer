"""
Bayesian Optimizer
==================
Optuna-backed Bayesian search for structured design extraction.
"""
import optuna
import numpy as np
from typing import Dict, Any

from ..generator.physics_guided import calculate_initial_bounds
from ..generator.random_generator import build_motor_dicts_from_vector
from ..propellants.openmotor_adapter import build_config, build_nozzle, build_propellant, assemble_motor
from ..simulation.openmotor_runner import run_simulation

class BayesianOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.constraints = config.get("constraints", {})
        self.mode = self.constraints.get("mode", "fast")
        self.bounds = calculate_initial_bounds(self.constraints, self.mode)
        
    def _objective(self, trial) -> float:
        # 1. Ask Optuna for values within bounds
        x = []
        for i, (low, high) in enumerate(self.bounds):
            if self.mode == "full" and i == 4:
                # Fin count must be int
                val = trial.suggest_int(f"x_{i}", int(low), int(high))
            else:
                val = trial.suggest_float(f"x_{i}", low, high)
            x.append(val)
            
        vector = np.array(x)
        
        nozzle_dict, grains_list = build_motor_dicts_from_vector(vector, self.constraints)
        
        if grains_list is None:
            # Optuna specifically supports prune exceptions
            raise optuna.TrialPruned()
            
        nozzle = build_nozzle(nozzle_dict["throat"], nozzle_dict["exit"])
        config = build_config(self.constraints)
        prop = build_propellant(self.config.get("propellant", {}))
        
        motor = assemble_motor(nozzle, config, prop, grains_list)
        success, metrics = run_simulation(motor)
        
        if not success:
            raise optuna.TrialPruned()
            
        target_mass = self.constraints.get("target_mass_kg", 0.1)
        mass_err = abs(metrics["propellant_mass_kg"] - target_mass) / target_mass
        
        pres_limit = self.constraints.get("max_pressure_Pa", 700 * 6894.757)
        if metrics["peak_pressure_Pa"] > pres_limit:
            raise optuna.TrialPruned()
            
        flux_limit = self.constraints.get("max_mass_flux", 500)
        if metrics["peak_mass_flux"] > flux_limit:
            raise optuna.TrialPruned()
            
        # If valid, just minimize mass error and maximize impulse
        impulse = metrics["total_impulse_Ns"]
        cost = (mass_err * 100) - (impulse * 0.05)
        
        # Save metrics as user attr
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        trial.set_user_attr("vector", vector.tolist())
            
        return cost

    def optimize(self) -> optuna.Study:
        print(f"Starting Bayesian Optimization with Optuna...")
        
        # Optuna logging config
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(direction="minimize")
        n_trials = self.config.get("optimizer", {}).get("samples", 200)
        
        study.optimize(self._objective, n_trials=n_trials)
        
        print(f"Bayesian Search completed.")
        return study
