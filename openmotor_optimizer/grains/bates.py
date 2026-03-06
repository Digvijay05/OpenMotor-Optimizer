"""
BATES Grain Configuration
=========================
A simple conceptual wrapper for BATES grain dimensions.
"""
from dataclasses import dataclass

@dataclass
class BatesConfig:
    outer_diameter: float
    length: float
    core_diameter: float
    
    def to_dict(self):
        """Converts to OpenMotor property dictionary format."""
        return {
            "type": "BATES",
            "diameter": self.outer_diameter,
            "length": self.length,
            "coreDiameter": self.core_diameter,
            "inhibitedEnds": "Neither" # Typical for BATES in this context
        }
