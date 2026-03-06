import os
import sys
import yaml
import time
import argparse
import pandas as pd
from datetime import datetime

from openmotor_optimizer.optimizer.montecarlo import MonteCarloOptimizer
from openmotor_optimizer.optimizer.genetic import GeneticOptimizer
from openmotor_optimizer.optimizer.bayesian import BayesianOptimizer
from openmotor_optimizer.generator.random_generator import build_motor_dicts_from_vector
from openmotor_optimizer.exporters.ric_exporter import export_to_ric

def main():
    parser = argparse.ArgumentParser(description="Run OpenMotor-Optimizer")
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    exp_name = config.get("experiment_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("experiments", "results", f"{exp_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Starting Experiment: {exp_name}")
    print(f"Results Directory: {out_dir}")
    print(f"{'='*50}\n")
    
    opt_type = config.get("optimizer", {}).get("type", "montecarlo").lower()
    
    df = None
    best_vector = None
    
    if opt_type == "montecarlo":
        optimizer = MonteCarloOptimizer(config)
        df = optimizer.optimize()
        if not df.empty and getattr(df, 'is_valid', None) is not None:
            # Sort by objective heuristically
            df = df.sort_values("total_impulse_Ns", ascending=False)
            best_vector = df.iloc[0]["vector"]
            
    elif opt_type == "genetic":
        optimizer = GeneticOptimizer(config)
        best_vector = optimizer.optimize()
        
    elif opt_type == "bayesian":
        optimizer = BayesianOptimizer(config)
        study = optimizer.optimize()
        best_vector = study.best_trial.user_attrs.get("vector")
        
    else:
        print(f"Unknown optimizer type: {opt_type}")
        return
        
    if df is not None and not df.empty:
        csv_path = os.path.join(out_dir, "motors.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} simulated motors to {csv_path}")
        
    if best_vector is not None:
        print("\nExtracting Best Design...")
        nozzle, grains = build_motor_dicts_from_vector(best_vector, config.get("constraints", {}))
        ric_path = os.path.join(out_dir, "best_motor.ric")
        export_to_ric(ric_path, config, grains, nozzle)
        print(f"Saved Best Profile to {ric_path}")
        
    print("\nExperiment Complete.")

if __name__ == "__main__":
    main()
