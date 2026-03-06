"""
GPU Batch Runner
================
Performs vectorized tensor operations (via PyTorch) to pre-evaluate and filter
massive parameter grids before feeding them to physical simulation.
"""
import torch
import numpy as np
import warnings

def filter_candidates_gpu(candidates: np.ndarray, config: dict, device_type: str = 'auto') -> np.ndarray:
    """
    Moves candidate matrix to compute device (GPU if available),
    applies rapid geometric heuristics, and returns surviving subset to CPU.
    """
    if device_type == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_type)
        
    t_cands = torch.tensor(candidates, dtype=torch.float32, device=device)
    od = config.get("grain_od_m", 0.035)
    mode = config.get("mode", "fast")
    
    try:
        if mode == "fast":
            # BATES Geometry Rules:
            # x[3] = bates_core
            valid_mask = t_cands[:, 3] < od
        else:
            # FINOCYL Geometry Rules:
            # x[3] = fino_core, x[6] = fin_len, x[8] = bates_core
            valid_mask = (t_cands[:, 3] < od) & (t_cands[:, 8] < od)
            
            # Fin reach cannot exceed outer radius
            fin_reach = (t_cands[:, 3] / 2.0) + t_cands[:, 6]
            valid_mask = valid_mask & (fin_reach <= (od / 2.0))
            
        filtered = t_cands[valid_mask]
        return filtered.cpu().numpy()
        
    except Exception as e:
        warnings.warn(f"GPU filtering failed, falling back to all candidates. Error: {e}")
        return candidates
