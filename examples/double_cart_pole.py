import mujoco

from hydrax.algs import PredictiveSampling, MPPI
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.double_cart_pole import DoubleCartPole
from hydrax.dynamics import (
    NeuralNetworkDynamics,
    SimplifiedPhysicsDynamics,
    DoubleCartPoleDynamics,
)
import argparse

"""
Run an interactive simulation of a double pendulum on a cart. Only the cart
is actuated, and the goal is to swing up the pendulum and balance it upright.
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Interactive double cart-pole simulation with different sampling algorithms"
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")

# Add dynamics model argument
parser.add_argument(
    "--dynamics",
    type=str,
    choices=["none", "nn", "deterministic"],
    default="none",
    help="Dynamics model to use for rollouts: 'none' (default MJX), 'nn' (neural network GRU), 'deterministic' (simplified physics)",
)

args = parser.parse_args()

# Create the base task without custom dynamics first
# We need the model to initialize custom dynamics
base_task = DoubleCartPole()

# Create custom dynamics model based on argument
custom_dynamics = None
if args.dynamics == "nn":
    print("Using Neural Network (GRU) dynamics for rollouts")
    custom_dynamics = NeuralNetworkDynamics(
        model=base_task.model,
        hidden_size=128,
        history_length=11,
    )
elif args.dynamics == "deterministic":
    print("Using Deterministic simplified physics dynamics for rollouts")
    custom_dynamics = DoubleCartPoleDynamics(
        model=base_task.model,
        integration_method="rk4",
        cart_mass=1.0,
        pole1_mass=0.1,
        pole2_mass=0.1,
        pole1_length=0.5,
        pole2_length=0.5,
        gravity=9.81,
    )
else:
    print("Using default MJX dynamics for rollouts")

# Define the task with custom dynamics
task = DoubleCartPole(custom_dynamics=custom_dynamics)

# Set up the controller
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=1024,
        noise_level=0.3,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=4,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=1024,
        noise_level=0.3,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=4,
    )

# Define the model used for simulation
mj_model = task.mj_model
mj_data = mujoco.MjData(mj_model)

# Run the interactive simulation
run_interactive(
    ctrl,
    mj_model,
    mj_data,
    frequency=50,
    fixed_camera_id=0,
    show_traces=True,
    max_traces=1,
)
