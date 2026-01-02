"""Example training script for the double cart pole task.

This script demonstrates how to train a neural network dynamics model
on data logged from MJX simulations of the double cart pole task.
"""

import jax.numpy as jnp
from pathlib import Path

from hydrax.dynamics_model_training.train import train_dynamics_model
from hydrax.dynamics_model_training.evaluate import plot_training_history


def main():
    """Train a dynamics model for the double cart pole task."""

    # Data directory (contains CSV files from MJX runs)
    data_dir = str(Path(__file__).parent.parent / 'data_csv' / 'double_cart_pole')

    # Task-specific configuration for double cart pole
    # The double cart pole has:
    #   - qpos: [cart_x, pole1_angle, pole2_angle]  (3 DOF)
    #   - qvel: [cart_x_vel, pole1_angle_vel, pole2_angle_vel]  (3 DOF)
    #   - ctrl: [cart_force]  (1 control input)

    # Coordinate indices: cart position (index 0) is a Cartesian coordinate
    coordinate_indices = jnp.array([0])

    # Angle indices: pole angles (indices 1 and 2) need cos/sin transformation
    angle_indices = jnp.array([1, 2])

    # Training hyperparameters
    config = {
        'data_dir': data_dir,
        'task_name': 'double_cart_pole',
        'coordinate_indices': coordinate_indices,
        'angle_indices': angle_indices,
        'hidden_size': 128,
        'history_length': 11,
        'prediction_horizon': 9,
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'seed': 42,
    }

    print("=" * 60)
    print("Training Dynamics Model for Double Cart Pole")
    print("=" * 60)
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Train the model
    network, history = train_dynamics_model(**config)

    # Plot training history
    checkpoint_dir = Path(__file__).parent / 'checkpoints' / config['task_name']
    plot_training_history(
        history,
        save_path=str(checkpoint_dir / 'training_history.png')
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {checkpoint_dir / 'best_model.pkl'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
