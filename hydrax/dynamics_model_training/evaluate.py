"""Evaluation utilities for trained dynamics models.

This module provides tools to evaluate trained dynamics models by:
    - Computing prediction errors over multiple horizons
    - Visualizing predictions vs ground truth
    - Comparing with MJX ground truth dynamics
"""

from pathlib import Path
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle

from hydrax.dynamics.nn_dynamics import GRUDynamicsNetwork
from hydrax.dynamics_model_training.data_loader import DynamicsDataset


def load_trained_model(checkpoint_path: str) -> Tuple[GRUDynamicsNetwork, Dict[str, Any]]:
    """Load a trained dynamics model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        Tuple of (network, config)
    """
    from pathlib import Path
    import orbax.checkpoint as ocp
    from flax import nnx

    checkpoint_path = Path(checkpoint_path)

    # Load config
    with open(checkpoint_path / 'config.pkl', 'rb') as f:
        config_data = pickle.load(f)

    config = config_data['config']

    # Convert lists back to arrays
    if config['coordinate_indices'] is not None:
        import jax.numpy as jnp
        config['coordinate_indices'] = jnp.array(config['coordinate_indices'])
    if config['angle_indices'] is not None:
        import jax.numpy as jnp
        config['angle_indices'] = jnp.array(config['angle_indices'])

    # Recreate network with same architecture
    network = GRUDynamicsNetwork(
        hidden_size=config['hidden_size'],
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        rngs=nnx.Rngs(0),
    )

    # Load state using Orbax
    checkpointer = ocp.PyTreeCheckpointer()
    graph_def, state = nnx.split(network)
    restored_state = checkpointer.restore(checkpoint_path / 'state', item=state)

    # Merge back
    network = nnx.merge(graph_def, restored_state)

    return network, config


def evaluate_model(
    network: GRUDynamicsNetwork,
    dataset: DynamicsDataset,
    config: Dict[str, Any],
    num_examples: int = 100,
) -> Dict[str, Any]:
    """Evaluate a trained dynamics model on a dataset.

    Args:
        network: Trained GRU dynamics network
        dataset: Dataset to evaluate on
        config: Model configuration
        num_examples: Number of examples to evaluate

    Returns:
        Dictionary containing evaluation metrics
    """
    from hydrax.dynamics_model_training.train import apply_delta_to_state, normalize_history

    hidden_size = config['hidden_size']
    transformed_qpos_dim = config['transformed_qpos_dim']
    history_length = config['history_length']
    prediction_horizon = config['prediction_horizon']
    sorted_angle_indices = dataset.sorted_angle_indices
    transformed_coord_indices = dataset.transformed_coord_indices

    # Metrics to track
    metrics = {
        'mse_per_step': [],  # MSE for each prediction step
        'mae_per_step': [],  # MAE for each prediction step
    }

    num_examples = min(num_examples, len(dataset))

    # Compute state dimension
    state_dim = transformed_qpos_dim + dataset.nv

    for i in range(num_examples):
        sequence = dataset[i]

        # Initialize state_history with first history_length timesteps
        state_history = sequence[:history_length]

        # Initialize GRU state
        initial_gru_state = jnp.zeros((hidden_size,))

        step_mse = []
        step_mae = []
        predicted_states = []

        for h in range(prediction_horizon):
            # Normalize current history's coordinate indices using first step as origin
            normalized_history = normalize_history(
                state_history,
                jnp.array(transformed_coord_indices),
                state_dim,
            )

            # Predict delta from normalized history (deterministic=True for evaluation)
            predicted_delta, _ = network(normalized_history, initial_gru_state, deterministic=True)

            # Get non-normalized last state in history (state only, not action)
            last_state = state_history[-1, :state_dim]

            # Add delta to non-normalized last state (using trig formulas for angles)
            next_state = apply_delta_to_state(
                last_state,
                predicted_delta,
                transformed_qpos_dim,
                sorted_angle_indices,
            )

            predicted_states.append(next_state)

            # Get ground truth state for this step
            ground_truth_state = sequence[history_length + h, :state_dim]

            # Compute metrics
            mse = jnp.mean(jnp.square(next_state - ground_truth_state))
            mae = jnp.mean(jnp.abs(next_state - ground_truth_state))

            step_mse.append(float(mse))
            step_mae.append(float(mae))

            # Update history for next step
            if h < prediction_horizon - 1:
                # Get next control from the sequence
                next_control = sequence[history_length + h, state_dim:]

                # Construct new state-action pair
                next_state_action = jnp.concatenate([next_state, next_control])

                # Update history: remove oldest, append new
                state_history = jnp.concatenate([
                    state_history[1:],
                    jnp.expand_dims(next_state_action, 0)
                ], axis=0)

        metrics['mse_per_step'].append(step_mse)
        metrics['mae_per_step'].append(step_mae)

    # Average over all examples
    metrics['avg_mse_per_step'] = jnp.mean(jnp.array(metrics['mse_per_step']), axis=0)
    metrics['avg_mae_per_step'] = jnp.mean(jnp.array(metrics['mae_per_step']), axis=0)

    return metrics


def plot_training_history(history: Dict[str, Any], save_path: str | None = None):
    """Plot training and validation loss curves.

    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training History', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_prediction_errors(metrics: Dict[str, Any], save_path: str | None = None):
    """Plot prediction errors over the prediction horizon.

    Args:
        metrics: Metrics dictionary from evaluate_model
        save_path: Optional path to save the plot
    """
    avg_mse = metrics['avg_mse_per_step']
    avg_mae = metrics['avg_mae_per_step']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # MSE plot
    ax1.plot(avg_mse, marker='o', linewidth=2, markersize=6)
    ax1.set_xlabel('Prediction Step', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('Mean Squared Error over Horizon', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # MAE plot
    ax2.plot(avg_mae, marker='o', linewidth=2, markersize=6, color='orange')
    ax2.set_xlabel('Prediction Step', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title('Mean Absolute Error over Horizon', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved prediction error plot to {save_path}")
    else:
        plt.show()

    plt.close()


def evaluate_from_checkpoint(
    checkpoint_path: str,
    data_dir: str,
    output_dir: str | None = None,
    num_examples: int = 100,
):
    """Load a trained model and evaluate it on a dataset.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing CSV data
        output_dir: Directory to save evaluation results and plots
        num_examples: Number of examples to evaluate
    """
    print(f"Loading model from {checkpoint_path}...")
    network, config = load_trained_model(checkpoint_path)

    print(f"Loading dataset from {data_dir}...")
    dataset = DynamicsDataset(
        data_dir=data_dir,
        history_length=config['history_length'],
        prediction_horizon=config['prediction_horizon'],
        coordinate_indices=config.get('coordinate_indices'),
        angle_indices=config.get('angle_indices'),
    )

    # Use validation split
    _, val_dataset = dataset.split(train_ratio=0.8)

    print(f"\nEvaluating model on {num_examples} examples...")
    metrics = evaluate_model(network, val_dataset, config, num_examples)

    # Print results
    print("\nEvaluation Results:")
    print(f"Average MSE per step:")
    for i, mse in enumerate(metrics['avg_mse_per_step']):
        print(f"  Step {i+1}: {mse:.6f}")

    print(f"\nAverage MAE per step:")
    for i, mae in enumerate(metrics['avg_mae_per_step']):
        print(f"  Step {i+1}: {mae:.6f}")

    # Save plots if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Plot training history if available
        if 'history' in config:
            plot_training_history(
                config['history'],
                save_path=str(output_dir / 'training_history.png')
            )

        # Plot prediction errors
        plot_prediction_errors(
            metrics,
            save_path=str(output_dir / 'prediction_errors.png')
        )

        # Save metrics as JSON
        import json
        metrics_serializable = {
            'avg_mse_per_step': [float(x) for x in metrics['avg_mse_per_step']],
            'avg_mae_per_step': [float(x) for x in metrics['avg_mae_per_step']],
        }

        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        print(f"\nResults saved to {output_dir}")

    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate a trained dynamics model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing CSV data')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save evaluation results')
    parser.add_argument('--num-examples', type=int, default=100,
                       help='Number of examples to evaluate')

    args = parser.parse_args()

    evaluate_from_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
    )
