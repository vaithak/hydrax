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


def normalize_history(
    history: jnp.ndarray,
    coordinate_indices: jnp.ndarray,
    state_dim: int,
) -> jnp.ndarray:
    """Normalize coordinate indices in history using first timestep as origin.

    Args:
        history: History of shape (history_length, state_dim + action_dim)
        coordinate_indices: Indices in the state to normalize (transformed coordinates)
        state_dim: Dimension of the state

    Returns:
        Normalized history with coordinates relative to first timestep
    """
    if len(coordinate_indices) == 0:
        return history

    # Extract first state's coordinates as origin
    origin = history[0, coordinate_indices]

    # Normalize all timesteps
    normalized_history = history.copy()
    for coord_idx in coordinate_indices:
        normalized_history = normalized_history.at[:, coord_idx].set(
            history[:, coord_idx] - origin
        )

    return normalized_history


def compute_autoregressive_loss(
    network: GRUDynamicsNetwork,
    sequence: jnp.ndarray,
    history_length: int,
    prediction_horizon: int,
    transformed_qpos_dim: int,
    sorted_angle_indices: list,
    transformed_coord_indices: list,
    hidden_size: int,
    state_dim: int,
) -> jnp.ndarray:
    """Compute autoregressive multi-step prediction loss.

    The model predicts delta states autoregressively:
        1. Initialize state_history with first history_length timesteps
        2. For each prediction step (0 to prediction_horizon-1):
           - Normalize current history's coordinate indices using first step as origin
           - Predict delta from normalized history
           - Add delta to non-normalized last state (using trig formulas for angles)
           - Construct new history by appending new state and removing oldest
        3. Compute loss as MSE between all predicted states and ground truth

    Args:
        network: The GRU dynamics network
        sequence: Complete sequence, shape (history_length + prediction_horizon, state_dim + action_dim)
        history_length: Number of timesteps in history
        prediction_horizon: Number of steps to predict
        transformed_qpos_dim: Dimension of transformed qpos
        sorted_angle_indices: Original indices of angles in qpos (sorted)
        transformed_coord_indices: Indices of coordinates in transformed state (for normalization)
        hidden_size: Size of GRU hidden state
        state_dim: Dimension of the state (transformed_qpos_dim + nv)

    Returns:
        Mean squared error loss over all prediction steps
    """
    action_dim = sequence.shape[1] - state_dim

    # Initialize state_history with first history_length timesteps
    state_history = sequence[:history_length]

    # Initialize GRU state
    initial_gru_state = jnp.zeros((hidden_size,))

    # Store predicted states for loss computation
    predicted_states = []

    for h in range(prediction_horizon):
        # Normalize current history's coordinate indices using first step as origin
        normalized_history = normalize_history(
            state_history,
            jnp.array(transformed_coord_indices),
            state_dim,
        )

        # Predict delta from normalized history (deterministic=False for training)
        predicted_delta, _ = network(normalized_history, initial_gru_state, deterministic=False)

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

        # Update history for next iteration (if not last step)
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

    # Stack predicted states
    predicted_states = jnp.stack(predicted_states, axis=0)  # (prediction_horizon, state_dim)

    # Get ground truth states from sequence
    ground_truth_states = sequence[history_length:history_length + prediction_horizon, :state_dim]

    # Compute loss as MSE between predicted and ground truth
    loss = jnp.mean(jnp.square(predicted_states - ground_truth_states))

    return loss


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
    sequence_sample = train_dataset[0]
    state_action_dim = sequence_sample.shape[1]
    transformed_qpos_dim = dataset.transformed_qpos_dim
    state_dim = transformed_qpos_dim + dataset.nv
    action_dim = state_action_dim - state_dim

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
        model=network,
        tx=optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
        wrt=nnx.Param,
    )

    # Get sorted angle indices and transformed coordinate indices for loss computation
    sorted_angle_indices = dataset.sorted_angle_indices
    transformed_coord_indices = dataset.transformed_coord_indices

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

    # Define training and validation step functions outside the loop
    @nnx.jit
    def train_step(model, batch_sequences):
        """JIT-compiled training step."""
        def loss_fn(model):
            def loss_fn_single(sequence):
                return compute_autoregressive_loss(
                    model,
                    sequence,
                    history_length,
                    prediction_horizon,
                    transformed_qpos_dim,
                    sorted_angle_indices,
                    transformed_coord_indices,
                    hidden_size,
                    state_dim,
                )

            loss_fn_batch = nnx.vmap(loss_fn_single)
            losses = loss_fn_batch(batch_sequences)
            return jnp.mean(losses)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        return loss, grads

    @nnx.jit
    def val_step(model, batch_sequences):
        """JIT-compiled validation step."""
        def loss_fn_single(sequence):
            return compute_autoregressive_loss(
                model,
                sequence,
                history_length,
                prediction_horizon,
                transformed_qpos_dim,
                sorted_angle_indices,
                transformed_coord_indices,
                hidden_size,
                state_dim,
            )

        loss_fn_batch = nnx.vmap(loss_fn_single)
        losses = loss_fn_batch(batch_sequences)
        return jnp.mean(losses)

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
                batch_sequences = train_dataset.get_batch(batch_idx)

                # Compute loss and gradients using JIT-compiled function
                loss, grads = train_step(network, batch_sequences)

                # Update parameters
                optimizer.update(network, grads)

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
            batch_sequences = val_dataset.get_batch(batch_idx)

            # Compute loss using JIT-compiled function
            loss = val_step(network, batch_sequences)
            val_losses.append(float(loss))

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Save checkpoint if best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = checkpoint_dir / 'best_model'

            # Save using Orbax
            from flax.training import orbax_utils
            import orbax.checkpoint as ocp
            import pickle

            # Split network into graph_def and state
            graph_def, state = nnx.split(network)

            # Create checkpointer
            checkpointer = ocp.PyTreeCheckpointer()

            # Save state using Orbax (force=True to allow overwriting)
            checkpointer.save(checkpoint_path / 'state', state, force=True)

            # Save config and history separately with pickle
            config_data = {
                'config': {
                    'hidden_size': hidden_size,
                    'state_dim': state_dim,
                    'action_dim': action_dim,
                    'history_length': history_length,
                    'prediction_horizon': prediction_horizon,
                    'coordinate_indices': coordinate_indices.tolist() if coordinate_indices is not None else None,
                    'angle_indices': angle_indices.tolist() if angle_indices is not None else None,
                    'transformed_qpos_dim': transformed_qpos_dim,
                    'normalize_velocities': dataset.normalize_velocities,
                    'qvel_min': dataset.qvel_min.tolist() if dataset.qvel_min is not None else None,
                    'qvel_max': dataset.qvel_max.tolist() if dataset.qvel_max is not None else None,
                    'qvel_range': dataset.qvel_range.tolist() if dataset.qvel_range is not None else None,
                },
                'history': history,
            }

            with open(checkpoint_path / 'config.pkl', 'wb') as f:
                pickle.dump(config_data, f)

            print(f"  → Saved best model to {checkpoint_path}")

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = checkpoint_dir / f'model_epoch_{epoch+1}'

            # Save using Orbax
            from flax.training import orbax_utils
            import orbax.checkpoint as ocp
            import pickle

            # Split network into graph_def and state
            graph_def, state = nnx.split(network)

            # Create checkpointer
            checkpointer = ocp.PyTreeCheckpointer()

            # Save state using Orbax (force=True to allow overwriting)
            checkpointer.save(checkpoint_path / 'state', state, force=True)

            # Save config and history separately with pickle
            config_data = {
                'config': {
                    'hidden_size': hidden_size,
                    'state_dim': state_dim,
                    'action_dim': action_dim,
                    'history_length': history_length,
                    'prediction_horizon': prediction_horizon,
                    'coordinate_indices': coordinate_indices.tolist() if coordinate_indices is not None else None,
                    'angle_indices': angle_indices.tolist() if angle_indices is not None else None,
                    'transformed_qpos_dim': transformed_qpos_dim,
                    'normalize_velocities': dataset.normalize_velocities,
                    'qvel_min': dataset.qvel_min.tolist() if dataset.qvel_min is not None else None,
                    'qvel_max': dataset.qvel_max.tolist() if dataset.qvel_max is not None else None,
                    'qvel_range': dataset.qvel_range.tolist() if dataset.qvel_range is not None else None,
                },
                'history': history,
            }

            with open(checkpoint_path / 'config.pkl', 'wb') as f:
                pickle.dump(config_data, f)

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {checkpoint_dir / 'best_model'}")

    return network, history
