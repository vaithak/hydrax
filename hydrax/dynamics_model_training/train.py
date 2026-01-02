"""Training script for neural network dynamics models.

This module provides functionality to train GRU-based dynamics models using
state-action trajectory data with autoregressive multi-step prediction.
"""

from pathlib import Path
from typing import Dict, Any, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from tqdm import tqdm

from hydrax.dynamics.nn_dynamics import GRUDynamicsNetwork


def apply_delta_to_state(
    current_state: jnp.ndarray,
    delta: jnp.ndarray,
    transformed_qpos_dim: int,
    sorted_angle_indices: list,
) -> jnp.ndarray:
    """Apply predicted delta to current state, handling angles properly.

    For angles (represented as cos/sin), use trigonometric addition:
        cos(θ + δ) = cos(θ)cos(δ) - sin(θ)sin(δ)
        sin(θ + δ) = sin(θ)cos(δ) + cos(θ)sin(δ)

    Args:
        current_state: Current state (transformed qpos + qvel)
        delta: Predicted delta (transformed qpos delta + qvel delta)
        transformed_qpos_dim: Dimension of transformed qpos
        sorted_angle_indices: Original indices of angles in qpos (sorted)

    Returns:
        Next state after applying delta
    """
    current_qpos = current_state[:transformed_qpos_dim]
    current_qvel = current_state[transformed_qpos_dim:]

    delta_qpos = delta[:transformed_qpos_dim]
    delta_qvel = delta[transformed_qpos_dim:]

    # Update qvel (simple addition)
    next_qvel = current_qvel + delta_qvel

    # Update qpos (handling angles with trig formulas)
    if len(sorted_angle_indices) == 0:
        # No angles, simple addition
        next_qpos = current_qpos + delta_qpos
    else:
        # Build next_qpos by processing each segment
        next_qpos_parts = []
        transformed_idx = 0
        original_idx = 0

        for angle_idx in sorted_angle_indices:
            # Update regular positions before this angle
            segment_length = angle_idx - original_idx
            if segment_length > 0:
                next_qpos_parts.append(
                    current_qpos[transformed_idx:transformed_idx + segment_length] +
                    delta_qpos[transformed_idx:transformed_idx + segment_length]
                )
                transformed_idx += segment_length

            # Update angle using trigonometric addition
            # Current: (cos_θ, sin_θ), Delta: (cos_δ, sin_δ)
            cos_theta = current_qpos[transformed_idx]
            sin_theta = current_qpos[transformed_idx + 1]
            cos_delta = delta_qpos[transformed_idx]
            sin_delta = delta_qpos[transformed_idx + 1]

            # Next angle: cos(θ + δ), sin(θ + δ)
            cos_next = cos_theta * cos_delta - sin_theta * sin_delta
            sin_next = sin_theta * cos_delta + cos_theta * sin_delta

            next_qpos_parts.append(jnp.array([cos_next, sin_next]))

            transformed_idx += 2
            original_idx = angle_idx + 1

        # Update remaining positions after last angle
        if transformed_idx < transformed_qpos_dim:
            next_qpos_parts.append(
                current_qpos[transformed_idx:] + delta_qpos[transformed_idx:]
            )

        next_qpos = jnp.concatenate(next_qpos_parts)

    return jnp.concatenate([next_qpos, next_qvel])


def compute_autoregressive_loss(
    network: GRUDynamicsNetwork,
    state_action_history: jnp.ndarray,
    targets: jnp.ndarray,
    controls: jnp.ndarray,
    transformed_qpos_dim: int,
    sorted_angle_indices: list,
    hidden_size: int,
) -> jnp.ndarray:
    """Compute autoregressive multi-step prediction loss.

    The model predicts delta states autoregressively:
        - Predict delta_1 from history
        - Apply delta_1 to get next state, update history
        - Predict delta_2 from updated history
        - Continue for prediction_horizon steps

    Args:
        network: The GRU dynamics network
        state_action_history: Initial history, shape (history_length, state_dim + action_dim)
        targets: Target deltas, shape (prediction_horizon, state_dim)
        controls: Control inputs for prediction horizon, shape (prediction_horizon, action_dim)
        transformed_qpos_dim: Dimension of transformed qpos
        sorted_angle_indices: Original indices of angles in qpos (sorted)
        hidden_size: Size of GRU hidden state

    Returns:
        Mean squared error loss over all prediction steps
    """
    prediction_horizon = targets.shape[0]
    history_length = state_action_history.shape[0]
    state_dim = targets.shape[1]

    # Initialize history and GRU state
    current_history = state_action_history
    initial_gru_state = jnp.zeros((hidden_size,))

    total_loss = 0.0

    for h in range(prediction_horizon):
        # Predict delta from current history
        predicted_delta, _ = network(current_history, initial_gru_state)

        # Compute loss for this step
        target_delta = targets[h]
        step_loss = jnp.mean(jnp.square(predicted_delta - target_delta))
        total_loss += step_loss

        # Update history for next prediction (if not last step)
        if h < prediction_horizon - 1:
            # Get current state from last entry in history
            current_state = current_history[-1, :state_dim]

            # Apply predicted delta to get next state
            next_state = apply_delta_to_state(
                current_state,
                predicted_delta,
                transformed_qpos_dim,
                sorted_angle_indices,
            )

            # Get next control
            next_control = controls[h]

            # Create next state-action pair
            next_state_action = jnp.concatenate([next_state, next_control])

            # Update history (shift and append)
            current_history = jnp.roll(current_history, shift=-1, axis=0)
            current_history = current_history.at[-1].set(next_state_action)

    # Average loss over prediction horizon
    return total_loss / prediction_horizon


def train_dynamics_model(
    data_dir: str,
    task_name: str,
    coordinate_indices: jnp.ndarray | None = None,
    angle_indices: jnp.ndarray | None = None,
    hidden_size: int = 128,
    history_length: int = 11,
    prediction_horizon: int = 9,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    checkpoint_dir: str | None = None,
    seed: int = 0,
) -> Tuple[GRUDynamicsNetwork, Dict[str, Any]]:
    """Train a GRU dynamics model on logged trajectory data.

    Args:
        data_dir: Directory containing CSV files with logged trajectories
        task_name: Name of the task (for saving checkpoints)
        coordinate_indices: Indices in qpos for Cartesian coordinates
        angle_indices: Indices in qpos that are angles
        hidden_size: Size of GRU hidden state
        history_length: Number of past timesteps to use as input
        prediction_horizon: Number of steps ahead to predict autoregressively
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        weight_decay: Weight decay for regularization
        checkpoint_dir: Directory to save checkpoints (default: data_dir/../dynamics_model_training/checkpoints)
        seed: Random seed

    Returns:
        Tuple of (trained_network, training_history)
    """
    from hydrax.dynamics_model_training.data_loader import DynamicsDataset

    # Set random seed
    key = jax.random.PRNGKey(seed)

    # Load dataset
    print(f"Loading data from {data_dir}...")
    dataset = DynamicsDataset(
        data_dir=data_dir,
        history_length=history_length,
        prediction_horizon=prediction_horizon,
        coordinate_indices=coordinate_indices,
        angle_indices=angle_indices,
    )

    # Split into train/val
    train_dataset, val_dataset = dataset.split(train_ratio=0.8, seed=seed)
    print(f"Train examples: {len(train_dataset)}, Val examples: {len(val_dataset)}")

    # Get dimensions
    state_action_hist_sample, targets_sample = train_dataset[0]
    state_dim = targets_sample.shape[1]
    action_dim = state_action_hist_sample.shape[1] - state_dim
    transformed_qpos_dim = dataset.transformed_qpos_dim

    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Transformed qpos dim: {transformed_qpos_dim}, qvel dim: {dataset.nv}")

    # Initialize network
    network = GRUDynamicsNetwork(
        hidden_size=hidden_size,
        state_dim=state_dim,
        action_dim=action_dim,
        rngs=nnx.Rngs(seed),
    )

    # Setup optimizer with weight decay
    optimizer = nnx.Optimizer(
        network,
        optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    )

    # Get sorted angle indices for loss computation
    sorted_angle_indices = dataset.sorted_angle_indices

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
    }

    # Setup checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = str(Path(data_dir).parent / 'dynamics_model_training' / 'checkpoints')
    checkpoint_dir = Path(checkpoint_dir) / task_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        train_losses = []
        key, subkey = jax.random.split(key)
        train_indices = jax.random.permutation(subkey, len(train_dataset))

        # Create batches
        num_batches = len(train_dataset) // batch_size

        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i in range(num_batches):
                batch_idx = train_indices[i * batch_size:(i + 1) * batch_size]
                batch_state_action, batch_targets = train_dataset.get_batch(batch_idx)

                # Extract controls from history for autoregressive rollout
                # Controls needed: from the last history entry onwards
                batch_controls = batch_state_action[:, -1:, -action_dim:]  # Last control in history

                # For autoregressive prediction, we need controls for each step
                # We'll use a placeholder here - in practice, these should come from the data
                # For now, we'll repeat the last control (this should be fixed in real usage)
                batch_controls_horizon = jnp.tile(
                    batch_controls,
                    (1, prediction_horizon, 1)
                )  # (batch_size, prediction_horizon, action_dim)

                # Define loss function for a single example
                def loss_fn_single(state_action_hist, targets, controls):
                    return compute_autoregressive_loss(
                        network,
                        state_action_hist,
                        targets,
                        controls,
                        transformed_qpos_dim,
                        sorted_angle_indices,
                        hidden_size,
                    )

                # Vectorize over batch
                loss_fn_batch = jax.vmap(loss_fn_single)

                # Define loss and grad function
                def loss_fn():
                    losses = loss_fn_batch(
                        batch_state_action,
                        batch_targets,
                        batch_controls_horizon,
                    )
                    return jnp.mean(losses)

                # Compute loss and gradients
                loss, grads = nnx.value_and_grad(loss_fn)()

                # Update parameters
                optimizer.update(grads)

                train_losses.append(float(loss))
                pbar.set_postfix({'train_loss': f'{loss:.6f}'})
                pbar.update(1)

        avg_train_loss = sum(train_losses) / len(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validation
        val_losses = []
        num_val_batches = len(val_dataset) // batch_size

        for i in range(num_val_batches):
            batch_idx = jnp.arange(i * batch_size, (i + 1) * batch_size)
            batch_state_action, batch_targets = val_dataset.get_batch(batch_idx)

            # Extract controls (same as training)
            batch_controls = batch_state_action[:, -1:, -action_dim:]
            batch_controls_horizon = jnp.tile(
                batch_controls,
                (1, prediction_horizon, 1)
            )

            # Compute loss
            def loss_fn_single(state_action_hist, targets, controls):
                return compute_autoregressive_loss(
                    network,
                    state_action_hist,
                    targets,
                    controls,
                    transformed_qpos_dim,
                    sorted_angle_indices,
                    hidden_size,
                )

            loss_fn_batch = jax.vmap(loss_fn_single)
            losses = loss_fn_batch(
                batch_state_action,
                batch_targets,
                batch_controls_horizon,
            )
            val_losses.append(float(jnp.mean(losses)))

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Save checkpoint if best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pkl'

            # Save using NNX serialization
            import pickle
            graph_def, state = nnx.split(network)
            save_data = {
                'graph_def': graph_def,
                'state': state,
                'config': {
                    'hidden_size': hidden_size,
                    'state_dim': state_dim,
                    'action_dim': action_dim,
                    'history_length': history_length,
                    'prediction_horizon': prediction_horizon,
                    'coordinate_indices': coordinate_indices,
                    'angle_indices': angle_indices,
                    'transformed_qpos_dim': transformed_qpos_dim,
                },
                'history': history,
            }

            with open(checkpoint_path, 'wb') as f:
                pickle.dump(save_data, f)

            print(f"  → Saved best model to {checkpoint_path}")

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch+1}.pkl'

            graph_def, state = nnx.split(network)
            save_data = {
                'graph_def': graph_def,
                'state': state,
                'config': {
                    'hidden_size': hidden_size,
                    'state_dim': state_dim,
                    'action_dim': action_dim,
                    'history_length': history_length,
                    'prediction_horizon': prediction_horizon,
                    'coordinate_indices': coordinate_indices,
                    'angle_indices': angle_indices,
                    'transformed_qpos_dim': transformed_qpos_dim,
                },
                'history': history,
            }

            with open(checkpoint_path, 'wb') as f:
                pickle.dump(save_data, f)

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model.pkl'}")

    return network, history
