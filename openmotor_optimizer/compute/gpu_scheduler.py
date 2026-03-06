"""
Compute Scheduler
=================
Intelligently assigns tensor sizes based on GPU/CPU availability.
"""
import torch
import multiprocessing

class ComputeScheduler:
    def __init__(self):
        self.has_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.has_gpu else 'cpu')
        self.cpu_cores = multiprocessing.cpu_count()
        
    def get_optimal_geometric_batch_size(self) -> int:
        """Returns how many random items to generate per block."""
        if self.has_gpu:
            return 100_000 # Easy for a GPU
        return 10_000    # CPU vectorized
        
    def get_physical_worker_count(self) -> int:
        """Returns optimal number of parallel Python processes for OpenMotor execution."""
        # Leave a core or two free for OS
        if self.cpu_cores > 2:
            return self.cpu_cores - 2
        return 1
