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
        checkpoint_path: Path to the checkpoint file

    Returns:
        Tuple of (network, config)
    """
    with open(checkpoint_path, 'rb') as f:
        save_data = pickle.load(f)

    # Reconstruct network
    network = jax.tree_util.tree_map(lambda x: x, save_data['graph_def'])
    network = jax.tree_util.tree_map(lambda x: x, save_data['state'])

    # Use nnx.merge to reconstruct the module
    from flax import nnx
    network = nnx.merge(save_data['graph_def'], save_data['state'])

    config = save_data['config']

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
    from hydrax.dynamics_model_training.train import apply_delta_to_state

    hidden_size = config['hidden_size']
    transformed_qpos_dim = config['transformed_qpos_dim']
    sorted_angle_indices = dataset.sorted_angle_indices

    # Metrics to track
    metrics = {
        'mse_per_step': [],  # MSE for each prediction step
        'mae_per_step': [],  # MAE for each prediction step
    }

    num_examples = min(num_examples, len(dataset))

    for i in range(num_examples):
        state_action_history, targets = dataset[i]
        prediction_horizon = targets.shape[0]
        state_dim = targets.shape[1]

        # Initialize
        current_history = state_action_history
        initial_gru_state = jnp.zeros((hidden_size,))

        step_mse = []
        step_mae = []

        for h in range(prediction_horizon):
            # Predict delta
            predicted_delta, _ = network(current_history, initial_gru_state)

            # Compare with target
            target_delta = targets[h]
            mse = jnp.mean(jnp.square(predicted_delta - target_delta))
            mae = jnp.mean(jnp.abs(predicted_delta - target_delta))

            step_mse.append(float(mse))
            step_mae.append(float(mae))

            # Update history for next step
            if h < prediction_horizon - 1:
                current_state = current_history[-1, :state_dim]
                next_state = apply_delta_to_state(
                    current_state,
                    predicted_delta,
                    transformed_qpos_dim,
                    sorted_angle_indices,
                )

                # Use zero control as placeholder (ideally should come from data)
                action_dim = current_history.shape[1] - state_dim
                next_control = current_history[-1, state_dim:]

                next_state_action = jnp.concatenate([next_state, next_control])
                current_history = jnp.roll(current_history, shift=-1, axis=0)
                current_history = current_history.at[-1].set(next_state_action)

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
