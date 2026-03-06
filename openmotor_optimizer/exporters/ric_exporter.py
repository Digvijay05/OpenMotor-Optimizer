"""
RIC Exporter
============
Generates and writes `.ric` files compatible with OpenMotor for visualization.
"""
import json
import os

def export_to_ric(filename: str, motor_config: dict, grains: list, nozzle: dict):
    """
    Exports a configuration to an OpenMotor .ric format.
    The format is typically JSON based.
    """
    # Create the internal structure expected by OpenMotor
    # (Since .ric files are usually JSON with specific nested structures)
    # The user has previously fought with .ric structures. We will construct a minimal valid representation.
    
    ric_data = {
        "formatVersion": 4,      # OpenMotor expects a format version, usually 4 or 5
        "units": "Metric",       # Or Imperial
        "propellant": motor_config.get("propellant", {}),
        "nozzle": {
            "throatDiameter": nozzle.get("throat", 0.0),
            "exitDiameter": nozzle.get("exit", 0.0),
            "efficiency": nozzle.get("efficiency", 0.85),
            "divertingAngle": nozzle.get("divAngle", 15.0),
            "convergingAngle": nozzle.get("convAngle", 45.0)
        },
        "grains": []
    }

    # Iterate grains and map to OpenMotor representation
    for grain in grains:
        gtype = grain.get("type", "BATES")
        props = {}
        if gtype == "BATES":
            props = {
                "type": "bates",
                "outerDiameter": grain.get("diameter", 0.0),
                "length": grain.get("length", 0.0),
                "coreDiameter": grain.get("coreDiameter", 0.0),
                "inhibitedEnds": grain.get("inhibitedEnds", "Neither")
            }
        elif gtype == "Finocyl":
            props = {
                "type": "finocyl",
                "outerDiameter": grain.get("diameter", 0.0),
                "length": grain.get("length", 0.0),
                "coreDiameter": grain.get("coreDiameter", 0.0),
                "numFins": grain.get("numFins", 4),
                "finWidth": grain.get("finWidth", 0.005),
                "finLength": grain.get("finLength", 0.010),
                "inhibitedEnds": grain.get("inhibitedEnds", "Neither")
            }
        ric_data["grains"].append(props)
        
    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(ric_data, f, indent=4)
        
    return os.path.exists(filename)
