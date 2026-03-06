"""
OpenMotor Adapter
=================
Translates OpenMotor-Optimizer dictionaries and concepts into OpenMotor object instances.
"""
import os
import sys
import warnings

# Attempt to load openMotor dynamically from environment variable or common relative path
OPENMOTOR_PATH = os.environ.get("OPENMOTOR_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../openMotor")))
if os.path.exists(OPENMOTOR_PATH) and OPENMOTOR_PATH not in sys.path:
    sys.path.insert(0, OPENMOTOR_PATH)

try:
    from motorlib.motor import Motor, MotorConfig
    from motorlib.nozzle import Nozzle
    from motorlib.propellant import Propellant
    from motorlib.grains.bates import BatesGrain
    from motorlib.grains.finocyl import Finocyl
except ImportError:
    warnings.warn(f"Could not import motorlib. Please set OPENMOTOR_PATH. Looked in {OPENMOTOR_PATH}")
    Motor, MotorConfig, Nozzle, Propellant, BatesGrain, Finocyl = None, None, None, None, None, None

def build_nozzle(throat_d: float, exit_d: float, properties: dict = None) -> 'Nozzle':
    """Builds an OpenMotor Nozzle."""
    nozzle = Nozzle()
    props = {
        "throat": throat_d,
        "exit": exit_d,
        "efficiency": 0.85,
        "divAngle": 15.0,
        "convAngle": 45.0,
        "throatLength": 0.0,
        "slagCoeff": 0.0,
        "erosionCoeff": 0.0
    }
    if properties:
        props.update(properties)
    nozzle.setProperties(props)
    return nozzle

def build_config(constraints: dict) -> 'MotorConfig':
    """Builds an OpenMotor MotorConfig from a constraints dict."""
    cfg = MotorConfig()
    props = {
        "maxPressure": constraints.get("max_pressure_Pa", 700 * 6894.757) * 1.5,
        "maxMassFlux": constraints.get("max_mass_flux", 500.0) * 1.5,
        "maxMachNumber": 0.95,
        "minPortThroat": 1.0,
        "flowSeparationWarnPercent": 0.5,
        "burnoutWebThres": 2.54e-5,
        "burnoutThrustThres": 0.1,
        "timestep": 0.003,
        "ambPressure": 101325.0,
        "mapDim": constraints.get("map_dim", 250),
        "sepPressureRatio": 0.3,
    }
    cfg.setProperties(props)
    return cfg

def build_propellant(prop_dict: dict) -> 'Propellant':
    """Builds an OpenMotor Propellant."""
    # Ensure backwards compatibility if prop_dict is already named
    return Propellant(prop_dict)

def create_bates_grain(od: float, length: float, core: float) -> 'BatesGrain':
    """Creates a basic Bates grain."""
    bg = BatesGrain()
    bg.setProperties({
        "diameter": od,
        "length": length,
        "coreDiameter": core,
        "inhibitedEnds": "Neither",
    })
    return bg

def create_finocyl_grain(od: float, length: float, core: float, num_fins: int, fin_w: float, fin_l: float) -> 'Finocyl':
    """Creates a basic Finocyl grain."""
    fino = Finocyl()
    fino.setProperties({
        "diameter": od,
        "length": length,
        "coreDiameter": core,
        "numFins": num_fins,
        "finWidth": fin_w,
        "finLength": fin_l,
        "invertedFins": False,
        "inhibitedEnds": "Neither",
    })
    return fino

def assemble_motor(nozzle: 'Nozzle', config: 'MotorConfig', prop: 'Propellant', grains: list) -> 'Motor':
    """Assembles a full OpenMotor Motor object."""
    motor = Motor()
    motor.propellant = prop
    motor.nozzle = nozzle
    motor.config = config
    
    for g in grains:
        if isinstance(g, dict):
            gtype = g.get("type", "BATES").upper()
            if gtype == "BATES":
                grain = create_bates_grain(
                    g.get("diameter", 0.0),
                    g.get("length", 0.0),
                    g.get("coreDiameter", 0.0)
                )
            elif gtype == "FINOCYL":
                grain = create_finocyl_grain(
                    g.get("diameter", 0.0),
                    g.get("length", 0.0),
                    g.get("coreDiameter", 0.0),
                    g.get("numFins", 4),
                    g.get("finWidth", 0.0),
                    g.get("finLength", 0.0)
                )
            else:
                raise ValueError(f"Unknown grain type: {gtype}")
            motor.grains.append(grain)
        else:
            motor.grains.append(g)

    # Initialize grain properties
    for grain in motor.grains:
        grain.simulationSetup(motor.config)

    return motor
