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

        # Create GRU cell and output layer
        input_dim = state_dim + action_dim
        self.gru_cell = nnx.GRUCell(input_dim, hidden_size, rngs=rngs)
        self.output_dense = nnx.Linear(hidden_size, state_dim, rngs=rngs)

    def __call__(self, state_action_history: jax.Array, gru_state: jax.Array):
        """Forward pass through the GRU dynamics model.

        Args:
            state_action_history: History of (state, action) pairs,
                                 shape (history_length, state_dim + action_dim)
            gru_state: Hidden state of the GRU, shape (hidden_size,)

        Returns:
            predicted_delta: Predicted change in state (delta_qpos, delta_qvel)
            new_gru_state: Updated GRU hidden state
        """
        # Process each timestep in the history through GRU
        carry = gru_state
        for t in range(state_action_history.shape[0]):
            carry, _ = self.gru_cell(carry, state_action_history[t])

        # Output layer to predict state delta
        delta = self.output_dense(carry)

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
        """
        self.model_ref = model
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.action_dim = model.nu

        # Store indices for state transformations
        self.coordinate_indices = coordinate_indices if coordinate_indices is not None else jnp.array([])
        self.angle_indices = angle_indices if angle_indices is not None else jnp.array([])

        # Validate that indices are within bounds
        # Note: jnp.any() on empty arrays returns False, so no need for length checks
        if jnp.any(self.angle_indices < 0) or jnp.any(self.angle_indices >= model.nq):
            raise ValueError(f"angle_indices must be in range [0, {model.nq})")
        if jnp.any(self.coordinate_indices < 0) or jnp.any(self.coordinate_indices >= model.nq):
            raise ValueError(f"coordinate_indices must be in range [0, {model.nq})")

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
        normalized = history  # JAX arrays are immutable, no need to copy

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

    def _update_qpos_from_delta(
        self, current_qpos: jax.Array, delta: jax.Array
    ) -> jax.Array:
        """Update qpos from predicted delta, handling angle updates properly.

        For angles: network predicts (cos(delta), sin(delta)). We extract the
        delta angle using arctan2, add it to the current angle, and normalize
        to [-pi, pi] range.

        Args:
            current_qpos: Original qpos (not transformed)
            delta: Predicted delta from network (in transformed space)

        Returns:
            Updated qpos in original space
        """
        if len(self.sorted_angle_indices) == 0:
            # No angles, simple addition
            return current_qpos + delta

        # Build updated qpos by processing each segment
        new_qpos = current_qpos  # JAX arrays are immutable, no need to copy
        transformed_idx = 0
        original_idx = 0

        # Use pre-sorted Python list to avoid JAX tracing issues
        for angle_idx in self.sorted_angle_indices:

            # Skip to this angle in the transformed space
            segment_length = angle_idx - original_idx
            transformed_idx += segment_length

            # Update regular positions before this angle
            if segment_length > 0:
                new_qpos = new_qpos.at[original_idx:angle_idx].add(
                    delta[transformed_idx - segment_length:transformed_idx]
                )

            # Update the angle
            # Delta is (cos(δ), sin(δ)) - extract angle delta using arctan2
            cos_delta = delta[transformed_idx]
            sin_delta = delta[transformed_idx + 1]
            angle_delta = jnp.arctan2(sin_delta, cos_delta)

            # Add delta to current angle and normalize to [-pi, pi]
            new_angle = current_qpos[angle_idx] + angle_delta
            # Normalize to [-pi, pi] range
            new_angle = jnp.arctan2(jnp.sin(new_angle), jnp.cos(new_angle))
            new_qpos = new_qpos.at[angle_idx].set(new_angle)

            transformed_idx += 2  # Skip the cos/sin pair
            original_idx = angle_idx + 1

        # Handle remaining positions after last angle
        if original_idx < len(current_qpos):
            # Add remaining delta values to remaining qpos values
            # Validate that dimensions match: each angle adds one extra dimension
            remaining_original = len(current_qpos) - original_idx
            remaining_transformed = len(delta) - transformed_idx
            if remaining_original != remaining_transformed:
                raise ValueError(
                    f"Dimension mismatch: {remaining_original} positions remain in qpos "
                    f"but {remaining_transformed} values remain in delta. This indicates "
                    f"an internal error in the transformation logic."
                )
            new_qpos = new_qpos.at[original_idx:].add(delta[transformed_idx:])

        return new_qpos

    def transform_state(self, data: mjx.Data) -> jax.Array:
        """Transform state to NN format (angles -> cos/sin).

        Args:
            data: Current state

        Returns:
            Transformed state with angles expanded to (cos, sin) pairs
        """
        transformed_qpos = self._transform_qpos_to_nn_format(data.qpos)
        return jnp.concatenate([transformed_qpos, data.qvel])

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

        # Predict state delta using NNX
        initial_gru_state = jnp.zeros((self.hidden_size,))
        delta, _ = self.network(normalized_history, initial_gru_state)

        # Split delta into qpos and qvel components
        delta_qpos = delta[:self.transformed_qpos_dim]
        delta_qvel = delta[self.transformed_qpos_dim:]

        # Update qpos (handling angles with arctan2 and normalization)
        new_qpos = self._update_qpos_from_delta(data.qpos, delta_qpos)

        # Update qvel (simple addition)
        new_qvel = data.qvel + delta_qvel

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
