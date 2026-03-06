"""
Monte Carlo Optimizer (GPU Accelerated)
=======================================
Massive random search using GPU pre-filtering.
"""
import time
import numpy as np
import multiprocessing
import pandas as pd
from typing import Dict, Any, Tuple

from ..generator.physics_guided import calculate_initial_bounds
from ..generator.random_generator import generate_random_candidates_cpu, build_motor_dicts_from_vector
from ..simulation.gpu_batch_runner import filter_candidates_gpu
from ..compute.gpu_scheduler import ComputeScheduler

from ..propellants.openmotor_adapter import build_config, build_nozzle, build_propellant, assemble_motor
from ..simulation.openmotor_runner import run_simulation

def _evaluate_single_vector(args: Tuple[np.ndarray, dict]) -> dict:
    """Worker function for multiprocessing pool."""
    vector, opt_config = args
    constraints = opt_config.get("constraints", {})
    
    nozzle_dict, grains_list = build_motor_dicts_from_vector(vector, constraints)
    
    if grains_list is None:
        return {"is_valid": False, "vector": vector.tolist()}
        
    try:
        nozzle = build_nozzle(nozzle_dict["throat"], nozzle_dict["exit"])
        config = build_config(constraints)
        prop = build_propellant(opt_config.get("propellant", {}))
        motor = assemble_motor(nozzle, config, prop, grains_list)
        
        success, metrics = run_simulation(motor)
        if success:
            metrics["is_valid"] = True
            metrics["vector"] = vector.tolist()
            return metrics
        else:
            return {"is_valid": False, "vector": vector.tolist()}
    except Exception:
        return {"is_valid": False, "vector": vector.tolist()}


class MonteCarloOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scheduler = ComputeScheduler()
        
    def optimize(self):
        constraints = self.config.get("constraints", {})
        mode = constraints.get("mode", "fast")
        total_samples = self.config.get("optimizer", {}).get("samples", 10000)
        
        print(f"Starting GPU-Accelerated Monte Carlo Search: {total_samples} samples.")
        
        bounds = calculate_initial_bounds(constraints, mode)
        
        # 1. Generate massive random matrix on CPU
        t0 = time.time()
        candidates = generate_random_candidates_cpu(bounds, total_samples)
        
        # 2. Filter invalid constraints rapidly on GPU
        filtered_candidates = filter_candidates_gpu(candidates, constraints, self.scheduler.device.type)
        survivors = len(filtered_candidates)
        print(f"GPU Filtering complete. {survivors}/{total_samples} passed geometric checks ({time.time()-t0:.2f}s).")
        
        # 3. Simulate survivors in parallel
        # Note: We must map the static config alongside the vector
        worker_count = self.scheduler.get_physical_worker_count()
        print(f"Spawning {worker_count} physical workers for OpenMotor simulation...")
        
        args_list = [(vec, self.config) for vec in filtered_candidates]
        
        results = []
        with multiprocessing.Pool(processes=worker_count) as pool:
            for i, res in enumerate(pool.imap_unordered(_evaluate_single_vector, args_list)):
                if res.get("is_valid", False):
                    results.append(res)
                if i > 0 and i % 100 == 0:
                    print(f"  Processed {i}/{survivors}...")
                    
        print(f"Simulation completed. {len(results)} valid motor profiles generated.")
        
        df = pd.DataFrame(results)
        return df
