"""Neural network-based dynamics model using GRU for state prediction.

This module implements a learned dynamics model that uses a GRU (Gated Recurrent Unit)
to predict the next state based on a history of (state, action) pairs.
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
            carry = self.gru_cell(state_action_history[t], carry)

        # Output layer to predict state delta
        delta = self.output_dense(carry)

        return delta, carry


class NeuralNetworkDynamics(DynamicsModel):
    """Neural network dynamics model using GRU for temporal modeling.

    This model maintains a history of the past 11 (state, action) pairs
    and uses a GRU to predict the next state autoregressively.
    """

    def __init__(
        self,
        model: mjx.Model,
        hidden_size: int = 128,
        history_length: int = 11,
        network: GRUDynamicsNetwork | None = None,
        model_path: str | None = None,
        seed: int = 0,
    ):
        """Initialize the neural network dynamics model.

        Args:
            model: The MJX model (for dimensions)
            hidden_size: Size of GRU hidden state
            history_length: Number of past (state, action) pairs to use
            network: Pre-initialized GRU network (if available)
            model_path: Path to saved model parameters (if available)
            seed: Random seed for initialization
        """
        self.model_ref = model
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.state_dim = model.nq + model.nv
        self.action_dim = model.nu

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

    def initialize_state_history(
        self, model: mjx.Model, data: mjx.Data | None = None
    ) -> jax.Array:
        """Initialize the state-action history buffer for GRU model.

        Args:
            model: The MJX model
            data: The current state (optional)

        Returns:
            Initialized history buffer of shape (history_length, state_action_dim)
        """
        state_dim = model.nq + model.nv
        action_dim = model.nu
        history_dim = state_dim + action_dim
        return jnp.zeros((self.history_length, history_dim))

    def step(
        self,
        model: mjx.Model,
        data: mjx.Data,
        state_history: jax.Array | None = None,
    ) -> Tuple[mjx.Data, jax.Array, jax.Array]:
        """Advance the state using the learned GRU dynamics model.

        Args:
            model: The MJX model
            data: Current state (with control in data.ctrl)
            state_history: History of (state, action) pairs

        Returns:
            next_data: Updated state
            updated_history: Updated state-action history
            updated_gru_state: Updated GRU hidden state
        """
        if state_history is None:
            state_history = self.initialize_state_history(model, data)

        # Update history with current state and action
        current_state_action = jnp.concatenate([data.qpos, data.qvel, data.ctrl])
        updated_history = jnp.roll(state_history, shift=-1, axis=0)
        updated_history = updated_history.at[-1].set(current_state_action)

        # Predict state delta using NNX
        # Always start with zero GRU state - each prediction is independent
        # The GRU processes the 11-step history internally, but we don't carry
        # the final GRU state between different predictions
        initial_gru_state = jnp.zeros((self.hidden_size,))
        delta, new_gru_state = self.network(updated_history, initial_gru_state)

        # Split delta into qpos and qvel components
        nq = model.nq
        delta_qpos = delta[:nq]
        delta_qvel = delta[nq:]

        # Update state
        new_qpos = data.qpos + delta_qpos
        new_qvel = data.qvel + delta_qvel

        # Create new data object with updated state
        next_data = data.replace(
            qpos=new_qpos,
            qvel=new_qvel,
            time=data.time + model.opt.timestep,
        )

        # Note: new_gru_state is returned but typically discarded by the caller
        # Each prediction starts fresh; GRU state is not carried between predictions
        return next_data, updated_history, new_gru_state

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            "type": "NeuralNetworkDynamics",
            "hidden_size": self.hidden_size,
            "history_length": self.history_length,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
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
        )

        return dynamics
