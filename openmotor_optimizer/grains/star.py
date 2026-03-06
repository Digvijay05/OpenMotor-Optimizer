"""
Star Grain Configuration
========================
A simple conceptual wrapper for Star grain dimensions.
"""
from dataclasses import dataclass

@dataclass
class StarConfig:
    outer_diameter: float
    length: float
    core_diameter: float
    num_points: int
    valley_rounding: float
    point_rounding: float
    angle: float
    
    def to_dict(self):
        """Converts to OpenMotor property dictionary format."""
        return {
            "type": "Star",
            "diameter": self.outer_diameter,
            "length": self.length,
            "coreDiameter": self.core_diameter,
            "numPoints": self.num_points,
            "valleyRounding": self.valley_rounding,
            "pointRounding": self.point_rounding,
            "angle": self.angle,
            "inhibitedEnds": "Neither"
        }
