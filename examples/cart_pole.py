import argparse

import mujoco
import jax.numpy as jnp
from pathlib import Path

from hydrax.algs import CEM, MPPI, PredictiveSampling, DIAL
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.cart_pole import CartPole
from hydrax.dynamics import NeuralNetworkDynamics
from hydrax import ROOT

"""
Run an interactive simulation of a cart-pole swingup
"""

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the cart-pole swingup task."
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
subparsers.add_parser("cem", help="Cross-Entropy Method")
subparsers.add_parser("dial", help="Diffusion-Inspired Annealing for Legged MPC")

# Add dynamics model argument
parser.add_argument(
    "--dynamics",
    type=str,
    choices=["none", "nn"],
    default="none",
    help="Dynamics model to use for rollouts: 'none' (default MJX), 'nn' (neural network GRU)",
)

# Add CSV logging argument
parser.add_argument(
    "--log_csv",
    action="store_true",
    help="Log state-action pairs to CSV for dynamics training"
)

args = parser.parse_args()

# Create the base task without custom dynamics first
# We need the model to initialize custom dynamics
base_task = CartPole()

# Create custom dynamics model based on argument
custom_dynamics = None
if args.dynamics == "nn":
    print("Using Neural Network (GRU) dynamics for rollouts")

    # Path to the best trained model checkpoint
    checkpoint_path = Path(ROOT) / 'data_csv' / 'dynamics_model_training' / 'checkpoints' / 'cart_pole' / 'best_model'

    if checkpoint_path.exists():
        print(f"Loading trained model from: {checkpoint_path}")
        custom_dynamics = NeuralNetworkDynamics.load_from_checkpoint(
            model=base_task.model,
            checkpoint_dir=str(checkpoint_path)
        )
        print("Successfully loaded trained dynamics model!")
    else:
        print(f"Warning: Trained checkpoint not found at {checkpoint_path}")
        print("Initializing untrained Neural Network (GRU) dynamics model instead")
        # Fallback to untrained model with same architecture as training
        custom_dynamics = NeuralNetworkDynamics(
            model=base_task.model,
            hidden_size=128,
            history_length=6,
            coordinate_indices=jnp.array([0]),
            angle_indices=jnp.array([1]),
        )
else:
    print("Using default MJX dynamics for rollouts")

# Define the task with custom dynamics
task = CartPole(custom_dynamics=custom_dynamics)

# Set up the controller
if args.algorithm == "ps" or args.algorithm is None:
    print("Running predictive sampling")
    ctrl = PredictiveSampling(
        task,
        num_samples=128,
        noise_level=0.3,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=4,
    )
elif args.algorithm == "mppi":
    print("Running MPPI")
    ctrl = MPPI(
        task,
        num_samples=128,
        noise_level=0.3,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=4,
    )
elif args.algorithm == "cem":
    print("Running CEM")
    ctrl = CEM(
        task,
        num_samples=128,
        num_elites=3,
        sigma_start=0.5,
        sigma_min=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=4,
    )
elif args.algorithm == "dial":
    print("Running DIAL")
    ctrl = DIAL(
        task,
        num_samples=128,
        noise_level=0.3,
        beta_opt_iter=1.0,
        beta_horizon=1.0,
        temperature=0.1,
        spline_type="cubic",
        plan_horizon=1.0,
        num_knots=4,
        iterations=3,
    )
else:
    parser.error("Other algorithms not implemented for this example!")

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
    show_traces=False,
    max_traces=1,
    log_csv=args.log_csv,
    task_name="cart_pole",
)
