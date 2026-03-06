"""
Finocyl Grain Configuration
===========================
A simple conceptual wrapper for Finocyl grain dimensions.
"""
from dataclasses import dataclass

@dataclass
class FinocylConfig:
    outer_diameter: float
    length: float
    core_diameter: float
    num_fins: int
    fin_width: float
    fin_length: float
    
    def to_dict(self):
        """Converts to OpenMotor property dictionary format."""
        return {
            "type": "Finocyl",
            "diameter": self.outer_diameter,
            "length": self.length,
            "coreDiameter": self.core_diameter,
            "numFins": self.num_fins,
            "finWidth": self.fin_width,
            "finLength": self.fin_length,
            "inhibitedEnds": "Neither"
        }
