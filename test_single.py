"""
Quick diagnostic: build ONE motor, simulate it, print everything.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from openmotor_optimizer.propellants.openmotor_adapter import build_nozzle, build_config, build_propellant, assemble_motor
from openmotor_optimizer.simulation.openmotor_runner import run_simulation
import numpy as np

# Known-good KNDX propellant
PROP = {
    "name": "KNDX",
    "density": 1879.0,
    "tabs": [
        {
            "minPressure": 100000,
            "maxPressure": 10300000,
            "a": 1.713e-6,
            "n": 0.619,
            "k": 1.1308,
            "t": 1710.0,
            "m": 42.39,
        }
    ],
}

CONSTRAINTS = {
    "target_mass_kg": 0.1,
    "max_pressure_Pa": 4826330,
    "max_mass_flux": 500,
    "grain_od_m": 0.035,
    "total_grains": 4,
    "mode": "fast",
}

# A hand-picked reasonable vector: throat=8mm, exit=14mm, grain_len=60mm, core=15mm
vector = np.array([0.008, 0.014, 0.060, 0.015])

# 1. Build grain dicts
from openmotor_optimizer.generator.random_generator import build_motor_dicts_from_vector
nozzle_dict, grains_list = build_motor_dicts_from_vector(vector, CONSTRAINTS)
print(f"Nozzle: {nozzle_dict}")
print(f"Grains ({len(grains_list)}): {grains_list[0]}")

# 2. Build motor
nozzle = build_nozzle(nozzle_dict["throat"], nozzle_dict["exit"])
config = build_config(CONSTRAINTS)
prop = build_propellant(PROP)
motor = assemble_motor(nozzle, config, prop, grains_list)

print(f"\nMotor grains: {len(motor.grains)}")
density = prop.getProperty("density")
mass = sum(g.getVolumeAtRegression(0) * density for g in motor.grains)
print(f"Propellant mass: {mass*1000:.2f} g")

# 3. Simulate
print("\nRunning simulation...")
success, metrics = run_simulation(motor)
print(f"Success: {success}")
if success:
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
else:
    print(f"Error: {metrics}")

# 4. Also check what channels exist
print("\nChecking sim channels directly...")
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sim = motor.runSimulation()
    print(f"  sim.success = {sim.success}")
    print(f"  Available channels: {list(sim.channels.keys())}")
    if hasattr(sim, 'getMaxForce'):
        print(f"  getMaxForce() = {sim.getMaxForce()}")
    print(f"  getMaxPressure() = {sim.getMaxPressure()}")
    print(f"  getImpulse() = {sim.getImpulse()}")
    print(f"  getAverageForce() = {sim.getAverageForce()}")
