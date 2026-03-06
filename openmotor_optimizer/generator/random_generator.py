"""
Random Geometry Generator
=========================
Fast random generator for Monte-Carlo operations using constraints and bounds.
Can eventually run on GPU via Numba or PyTorch for massive batches.
"""
import numpy as np
from typing import Dict, List, Tuple
from ..grains.bates import BatesConfig
from ..grains.finocyl import FinocylConfig

def generate_random_candidates_cpu(bounds: List[Tuple[float, float]], n_samples: int) -> np.ndarray:
    """
    Generates N random candidates within bounds uniformly.
    """
    dims = len(bounds)
    samples = np.random.rand(n_samples, dims)
    
    for i in range(dims):
        low, high = bounds[i]
        samples[:, i] = low + samples[:, i] * (high - low)
        
    return samples

def build_motor_dicts_from_vector(vector: np.ndarray, config: dict) -> Tuple[dict, list]:
    """
    Translates a 1D vector back into a nozzle and grain list.
    Supports parsing BATES and Finocyl modes depending on length of vector.
    """
    mode = config.get("mode", "fast")
    od = config.get("grain_od_m", 0.035)
    total_g = config.get("total_grains", 4)
    
    throat_d = vector[0]
    exit_d = max(vector[1], throat_d + 1e-4) # enforce exit > throat
    
    nozzle = {
        "throat": throat_d,
        "exit": exit_d
    }
    grains = []
    
    if mode == "fast":
        # BATES ONLY
        bates_len = vector[2]
        bates_core = vector[3]
        
        # Geometrical constraint check
        if bates_core >= od:
            return nozzle, None # Invalid
            
        for _ in range(total_g):
            grains.append(BatesConfig(od, bates_len, bates_core).to_dict())
            
    elif mode == "full":
        # 1 Finocyl + N BATES
        fino_len = vector[2]
        fino_core = vector[3]
        num_fins = int(round(vector[4]))
        fin_w = vector[5]
        fin_l = vector[6]
        bates_len = vector[7]
        bates_core = vector[8]
        
        if fino_core >= od or bates_core >= od:
            return nozzle, None
        if (fino_core / 2.0) + fin_l > (od / 2.0):
            return nozzle, None
            
        grains.append(FinocylConfig(od, fino_len, fino_core, num_fins, fin_w, fin_l).to_dict())
        for _ in range(total_g - 1):
            grains.append(BatesConfig(od, bates_len, bates_core).to_dict())
            
    return nozzle, grains
