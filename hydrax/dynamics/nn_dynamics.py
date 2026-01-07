"""Neural network-based dynamics model using GRU for state prediction.

This module implements a learned dynamics model that uses a GRU (Gated Recurrent Unit)
to predict the next state based on a history of (state, action) pairs.

The model supports:
- Coordinate normalization: Cartesian coordinates are normalized relative to the first
  point in history when passed to the network (but stored unnormalized in history)
- Angle representation: Angles are converted to (cos, sin) in history and the network
  predicts (cos(delta), sin(delta)) for updates
"""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from mujoco import mjx

from hydrax.dynamics_base import DynamicsModel


class GRUDynamicsNetwork(nnx.Module):
    """GRU-based neural network for dynamics prediction.

    Architecture:
        Input: (state, action) pairs from history window
        GRU: Processes temporal sequence
        Dense: Maps GRU output to next state prediction (delta)
    """

    def __init__(
        self,
        hidden_size: int,
        state_dim: int,
        action_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize the GRU dynamics network.

        Args:
            hidden_size: Size of GRU hidden state
            state_dim: Dimension of state (nq + nv)
            action_dim: Dimension of action (nu)
            rngs: Random number generator state
        """
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Create GRU cell and output layers with dropout
        input_dim = state_dim + action_dim
        self.gru_cell = nnx.GRUCell(input_dim, hidden_size, rngs=rngs)
        self.output_dense1 = nnx.Linear(hidden_size, hidden_size//2, rngs=rngs)
        # self.dropout1 = nnx.Dropout(rate=0.1, rngs=rngs)
        self.output_dense2 = nnx.Linear(hidden_size//2, state_dim, rngs=rngs)

    def __call__(self, state_action_history: jax.Array, gru_state: jax.Array, deterministic: bool = False):
        """Forward pass through the GRU dynamics model.

        Args:
            state_action_history: History of (state, action) pairs,
                                 shape (history_length, state_dim + action_dim)
            gru_state: Hidden state of the GRU, shape (hidden_size,)
            deterministic: If True, disable dropout (for evaluation)

        Returns:
            predicted_delta: Predicted change in state (delta_qpos, delta_qvel)
            new_gru_state: Updated GRU hidden state
        """
        # Process each timestep in the history through GRU
        carry = gru_state
        for t in range(state_action_history.shape[0]):
            carry, _ = self.gru_cell(carry, state_action_history[t])

        # Two-layer output with ReLU and dropout
        x = self.output_dense1(carry)
        x = nnx.relu(x)
        # x = self.dropout1(x, deterministic=deterministic)
        delta = self.output_dense2(x)

        return delta, carry


class NeuralNetworkDynamics(DynamicsModel):
    """Neural network dynamics model using GRU for temporal modeling.

    This model maintains a history of the past 11 (state, action) pairs
    and uses a GRU to predict the next state autoregressively.

    Supports coordinate normalization and angle representation for better
    neural network performance.
    """

    def __init__(
        self,
        model: mjx.Model,
        hidden_size: int = 128,
        history_length: int = 11,
        network: GRUDynamicsNetwork | None = None,
        model_path: str | None = None,
        seed: int = 0,
        coordinate_indices: jax.Array | None = None,
        angle_indices: jax.Array | None = None,
        normalize_velocities: bool = False,
        qvel_min: jax.Array | None = None,
        qvel_max: jax.Array | None = None,
        qvel_range: jax.Array | None = None,
    ):
        """Initialize the neural network dynamics model.

        Args:
            model: The MJX model (for dimensions)
            hidden_size: Size of GRU hidden state
            history_length: Number of past (state, action) pairs to use
            network: Pre-initialized GRU network (if available)
            model_path: Path to saved model parameters (if available)
            seed: Random seed for initialization
            coordinate_indices: Indices in qpos for Cartesian coordinates (for normalization)
            angle_indices: Indices in qpos for angles (converted to cos/sin)
            normalize_velocities: If True, velocities are normalized to [-1, 1]
            qvel_min: Min velocity values for normalization (required if normalize_velocities=True)
            qvel_max: Max velocity values for normalization (required if normalize_velocities=True)
            qvel_range: Velocity range for normalization (required if normalize_velocities=True)
        """
        self.model_ref = model
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.action_dim = model.nu

        # Store indices for state transformations
        self.coordinate_indices = coordinate_indices if coordinate_indices is not None else jnp.array([])
        self.angle_indices = angle_indices if angle_indices is not None else jnp.array([])

        # Store velocity normalization parameters
        self.normalize_velocities = normalize_velocities
        self.qvel_min = qvel_min
        self.qvel_max = qvel_max
        self.qvel_range = qvel_range

        # Pre-sort angle indices and convert to Python list for iteration
        # This avoids JAX tracing issues when iterating over angle indices
        if len(self.angle_indices) > 0:
            self.sorted_angle_indices = [int(idx) for idx in jnp.sort(self.angle_indices)]
        else:
            self.sorted_angle_indices = []

        # Calculate transformed state dimension
        # Each angle becomes 2 values (cos, sin), other positions stay the same
        nq = model.nq
        nv = model.nv
        num_angles = len(self.angle_indices)
        self.transformed_qpos_dim = nq + num_angles  # nq positions + num_angles extra for cos/sin
        self.state_dim = self.transformed_qpos_dim + nv  # transformed qpos + qvel

        # Initialize or load the network
        if network is None:
            if model_path is not None:
                # TODO: Load from file
                raise NotImplementedError("Loading from file not yet implemented")
            else:
                # Initialize with random parameters using NNX
                self.network = GRUDynamicsNetwork(
                    hidden_size=hidden_size,
                    state_dim=self.state_dim,
                    action_dim=self.action_dim,
                    rngs=nnx.Rngs(seed),
                )
        else:
            self.network = network

        # Precompute transformed coordinate indices for efficiency
        self.transformed_coord_indices = self._compute_transformed_coord_indices()

        # Convert to Python list for iteration to avoid JAX tracing issues
        if len(self.transformed_coord_indices) > 0:
            self.transformed_coord_indices_list = [int(idx) for idx in self.transformed_coord_indices]
        else:
            self.transformed_coord_indices_list = []

    def _compute_transformed_coord_indices(self) -> jax.Array:
        """Compute coordinate indices after angle expansion."""
        if len(self.coordinate_indices) == 0:
            return jnp.array([])

        transformed_indices = []
        for coord_idx in self.coordinate_indices:
            # Each angle before this index adds one extra dimension
            num_angles_before = jnp.sum(self.angle_indices < coord_idx)
            transformed_idx = coord_idx + num_angles_before
            transformed_indices.append(transformed_idx)

        return jnp.array(transformed_indices)

    def _transform_qpos_to_nn_format(self, qpos: jax.Array) -> jax.Array:
        """Transform qpos to neural network format (angles -> cos/sin).

        Args:
            qpos: Original qpos array

        Returns:
            Transformed qpos with angles expanded to (cos, sin)
        """
        if len(self.sorted_angle_indices) == 0:
            return qpos

        # Build transformed array by processing segments between angles
        parts = []
        prev_idx = 0

        # Iterate over pre-sorted Python list to avoid JAX tracing issues
        for angle_idx in self.sorted_angle_indices:
            # Add regular values before this angle
            if angle_idx > prev_idx:
                parts.append(qpos[prev_idx:angle_idx])
            # Add cos/sin for the angle
            parts.append(jnp.array([jnp.cos(qpos[angle_idx]), jnp.sin(qpos[angle_idx])]))
            prev_idx = angle_idx + 1

        # Add any remaining regular values
        if prev_idx < len(qpos):
            parts.append(qpos[prev_idx:])

        return jnp.concatenate(parts)

    def _normalize_coordinates(self, history: jax.Array) -> jax.Array:
        """Normalize coordinates relative to first timestep (for network input only).

        Args:
            history: State-action history with transformed qpos

        Returns:
            History with normalized coordinates
        """
        if len(self.transformed_coord_indices_list) == 0:
            return history

        # Subtract first frame's coordinate values from all frames
        reference_qpos = history[0, :self.transformed_qpos_dim]
        normalized = history.copy()

        # Use pre-converted Python list to avoid JAX tracing issues
        for idx in self.transformed_coord_indices_list:
            normalized = normalized.at[:, idx].add(-reference_qpos[idx])

        return normalized

    def initialize_state_history(
        self, model: mjx.Model, data: mjx.Data | None = None
    ) -> jax.Array:
        """Initialize the state-action history buffer for GRU model.

        Args:
            model: The MJX model
            data: The current state (optional)

        Returns:
            Initialized history buffer of shape (history_length, transformed_state_dim + action_dim)
        """
        action_dim = model.nu
        history_dim = self.state_dim + action_dim
        return jnp.zeros((self.history_length, history_dim))

    def _apply_delta_to_transformed_qpos(
        self, current_qpos_transformed: jax.Array, delta: jax.Array
    ) -> jax.Array:
        """Apply delta to transformed qpos using trigonometric addition for angles.

        This matches the training logic where deltas are applied in transformed space.
        For angles represented as (cos, sin), use trig addition formulas:
            cos(θ + δ) = cos(θ)cos(δ) - sin(θ)sin(δ)
            sin(θ + δ) = sin(θ)cos(δ) + cos(θ)sin(δ)

        Args:
            current_qpos_transformed: Current qpos in transformed space (angles as cos/sin)
            delta: Predicted delta in transformed space

        Returns:
            Updated qpos in transformed space
        """
        if len(self.sorted_angle_indices) == 0:
            # No angles, simple addition
            return current_qpos_transformed + delta

        # Build next_qpos by processing each segment
        next_qpos_parts = []
        transformed_idx = 0
        original_idx = 0

        for angle_idx in self.sorted_angle_indices:
            # Update regular positions before this angle
            segment_length = angle_idx - original_idx
            if segment_length > 0:
                next_qpos_parts.append(
                    current_qpos_transformed[transformed_idx:transformed_idx + segment_length] +
                    delta[transformed_idx:transformed_idx + segment_length]
                )
                transformed_idx += segment_length

            # Update angle using trigonometric addition
            # Current: (cos_θ, sin_θ), Delta: (cos_δ, sin_δ)
            cos_theta = current_qpos_transformed[transformed_idx]
            sin_theta = current_qpos_transformed[transformed_idx + 1]
            cos_delta = delta[transformed_idx]
            sin_delta = delta[transformed_idx + 1]

            # Next angle: cos(θ + δ), sin(θ + δ)
            cos_next = cos_theta * cos_delta - sin_theta * sin_delta
            sin_next = sin_theta * cos_delta + cos_theta * sin_delta

            next_qpos_parts.append(jnp.array([cos_next, sin_next]))

            transformed_idx += 2
            original_idx = angle_idx + 1

        # Update remaining positions after last angle
        if transformed_idx < len(current_qpos_transformed):
            next_qpos_parts.append(
                current_qpos_transformed[transformed_idx:] + delta[transformed_idx:]
            )

        return jnp.concatenate(next_qpos_parts)

    def _transform_qpos_to_original(self, qpos_transformed: jax.Array) -> jax.Array:
        """Convert transformed qpos (with cos/sin for angles) back to original space.

        Args:
            qpos_transformed: qpos in transformed space (angles as cos/sin pairs)

        Returns:
            qpos in original space (angles as single values)
        """
        if len(self.sorted_angle_indices) == 0:
            return qpos_transformed

        # Build original qpos by converting cos/sin back to angles
        parts = []
        transformed_idx = 0
        original_idx = 0

        for angle_idx in self.sorted_angle_indices:
            # Add regular positions before this angle
            segment_length = angle_idx - original_idx
            if segment_length > 0:
                parts.append(qpos_transformed[transformed_idx:transformed_idx + segment_length])
                transformed_idx += segment_length

            # Convert (cos, sin) back to angle using arctan2
            cos_val = qpos_transformed[transformed_idx]
            sin_val = qpos_transformed[transformed_idx + 1]
            angle = jnp.arctan2(sin_val, cos_val)
            parts.append(jnp.array([angle]))

            transformed_idx += 2
            original_idx = angle_idx + 1

        # Add remaining positions after last angle
        if transformed_idx < len(qpos_transformed):
            parts.append(qpos_transformed[transformed_idx:])

        return jnp.concatenate(parts)

    def _normalize_qvel(self, qvel: jax.Array) -> jax.Array:
        """Normalize velocities to [-1, 1] range.

        Args:
            qvel: Velocity array

        Returns:
            Normalized velocity array
        """
        if not self.normalize_velocities:
            return qvel

        # Scale to [-1, 1]: (qvel - min) / (max - min) * 2 - 1
        normalized = (qvel - self.qvel_min) / self.qvel_range * 2.0 - 1.0
        return normalized

    def _denormalize_qvel(self, qvel_normalized: jax.Array) -> jax.Array:
        """Denormalize velocities from [-1, 1] range back to original scale.

        Args:
            qvel_normalized: Normalized velocity array

        Returns:
            Denormalized velocity array
        """
        if not self.normalize_velocities:
            return qvel_normalized

        # Inverse scaling: (qvel_norm + 1) / 2 * (max - min) + min
        denormalized = (qvel_normalized + 1.0) / 2.0 * self.qvel_range + self.qvel_min
        return denormalized

    def transform_state(self, data: mjx.Data) -> jax.Array:
        """Transform state to NN format (angles -> cos/sin, velocities normalized).

        Args:
            data: Current state

        Returns:
            Transformed state with angles expanded to (cos, sin) pairs and normalized velocities
        """
        transformed_qpos = self._transform_qpos_to_nn_format(data.qpos)
        normalized_qvel = self._normalize_qvel(data.qvel)
        return jnp.concatenate([transformed_qpos, normalized_qvel])

    def step(
        self,
        model: mjx.Model,
        data: mjx.Data,
        state_history: jax.Array | None = None,
    ) -> Tuple[mjx.Data, jax.Array]:
        """Advance the state using the learned GRU dynamics model.

        Args:
            model: The MJX model
            data: Current state (with control in data.ctrl)
            state_history: History of (state, action) pairs (stores transformed states)

        Returns:
            next_data: Updated state
            updated_history: Updated state-action history (needed for autoregressive rollouts)
        """
        if state_history is None:
            state_history = self.initialize_state_history(model, data)

        # Update history with current state and action using base class method
        # This automatically uses transform_state to get the correct format
        updated_history = self.update_state_history(state_history, data, data.ctrl)

        # Normalize coordinates for network input (only positions, not velocities)
        normalized_history = self._normalize_coordinates(updated_history)

        # Predict state delta using NNX (deterministic=True for inference)
        initial_gru_state = jnp.zeros((self.hidden_size,))
        delta, _ = self.network(normalized_history, initial_gru_state, deterministic=True)

        # Split delta into qpos and qvel components
        delta_qpos = delta[:self.transformed_qpos_dim]
        delta_qvel_normalized = delta[self.transformed_qpos_dim:]

        # Get the last state from history (in transformed space with normalized qvel)
        # This is the state the network used to make the prediction
        last_state_transformed = updated_history[-1, :self.transformed_qpos_dim + model.nv]
        last_qpos_transformed = last_state_transformed[:self.transformed_qpos_dim]
        last_qvel_normalized = last_state_transformed[self.transformed_qpos_dim:]

        # Apply delta to transformed qpos using trigonometric addition for angles
        new_qpos_transformed = self._apply_delta_to_transformed_qpos(last_qpos_transformed, delta_qpos)

        # Convert transformed qpos back to original space
        new_qpos = self._transform_qpos_to_original(new_qpos_transformed)

        # Update qvel in normalized space and denormalize
        new_qvel_normalized = last_qvel_normalized + delta_qvel_normalized
        new_qvel = self._denormalize_qvel(new_qvel_normalized)

        # Create new data object with updated state
        next_data = data.replace(
            qpos=new_qpos,
            qvel=new_qvel,
            time=data.time + model.opt.timestep,
        )

        return next_data, updated_history

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            "type": "NeuralNetworkDynamics",
            "hidden_size": self.hidden_size,
            "history_length": self.history_length,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "coordinate_indices": self.coordinate_indices,
            "angle_indices": self.angle_indices,
            "normalize_velocities": self.normalize_velocities,
            "qvel_min": self.qvel_min,
            "qvel_max": self.qvel_max,
            "qvel_range": self.qvel_range,
        }

    def save_params(self, path: str):
        """Save model parameters to disk using NNX serialization.

        Args:
            path: Path to save the parameters
        """
        import pickle
        from pathlib import Path

        # Extract the state from the NNX model
        graph_def, state = nnx.split(self.network)

        # Save both the graph definition and state
        save_data = {
            "graph_def": graph_def,
            "state": state,
            "config": self.get_config(),
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    @classmethod
    def load_params(cls, model: mjx.Model, path: str) -> "NeuralNetworkDynamics":
        """Load model parameters from disk using NNX deserialization.

        Args:
            model: The MJX model
            path: Path to the saved parameters

        Returns:
            Initialized NeuralNetworkDynamics with loaded parameters
        """
        import pickle

        with open(path, "rb") as f:
            save_data = pickle.load(f)

        # Reconstruct the network from graph_def and state
        network = nnx.merge(save_data["graph_def"], save_data["state"])

        # Create the dynamics model with the loaded network
        config = save_data["config"]
        dynamics = cls(
            model=model,
            hidden_size=config["hidden_size"],
            history_length=config["history_length"],
            network=network,
            coordinate_indices=config.get("coordinate_indices"),
            angle_indices=config.get("angle_indices"),
        )

        return dynamics

    @classmethod
    def load_from_checkpoint(cls, model: mjx.Model, checkpoint_dir: str) -> "NeuralNetworkDynamics":
        """Load model from a training checkpoint directory (Orbax format).

        This method loads checkpoints saved by the training script, which uses
        Orbax for state serialization and pickle for config.

        Args:
            model: The MJX model
            checkpoint_dir: Path to checkpoint directory (e.g., 'checkpoints/double_cart_pole/best_model')

        Returns:
            Initialized NeuralNetworkDynamics with loaded parameters
        """
        import pickle
        import orbax.checkpoint as ocp
        from pathlib import Path

        checkpoint_path = Path(checkpoint_dir)

        # Load config
        with open(checkpoint_path / 'config.pkl', 'rb') as f:
            config_data = pickle.load(f)

        config = config_data['config']

        # Convert coordinate/angle indices to JAX arrays
        coordinate_indices = jnp.array(config['coordinate_indices']) if config['coordinate_indices'] is not None else None
        angle_indices = jnp.array(config['angle_indices']) if config['angle_indices'] is not None else None

        # Load velocity normalization parameters (with backward compatibility)
        normalize_velocities = config.get('normalize_velocities', False)
        qvel_min = jnp.array(config['qvel_min']) if config.get('qvel_min') is not None else None
        qvel_max = jnp.array(config['qvel_max']) if config.get('qvel_max') is not None else None
        qvel_range = jnp.array(config['qvel_range']) if config.get('qvel_range') is not None else None

        # Initialize network with same architecture
        network = GRUDynamicsNetwork(
            hidden_size=config['hidden_size'],
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            rngs=nnx.Rngs(0),  # Seed doesn't matter, we'll overwrite params
        )

        # Load state using Orbax with StandardRestore to handle sharding
        checkpointer = ocp.StandardCheckpointer()

        # Restore the state
        state = checkpointer.restore(checkpoint_path / 'state')

        # Merge graph_def and loaded state
        graph_def, _ = nnx.split(network)
        network = nnx.merge(graph_def, state)

        # Create the dynamics model with the loaded network
        dynamics = cls(
            model=model,
            hidden_size=config['hidden_size'],
            history_length=config['history_length'],
            network=network,
            coordinate_indices=coordinate_indices,
            angle_indices=angle_indices,
            normalize_velocities=normalize_velocities,
            qvel_min=qvel_min,
            qvel_max=qvel_max,
            qvel_range=qvel_range,
        )

        return dynamics
