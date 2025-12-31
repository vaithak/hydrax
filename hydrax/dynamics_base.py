"""Abstract base class for custom dynamics models.

This module provides an interface for implementing custom dynamics models
that can be used for rollouts in place of the default MJX physics simulator.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import jax
import jax.numpy as jnp
from mujoco import mjx


class DynamicsModel(ABC):
    """Abstract base class for custom dynamics models.

    This interface allows you to replace the default MJX physics simulator
    with a custom dynamics model for controller rollouts. The simulator
    model remains unchanged - only the model used for planning is affected.

    Custom dynamics models can be:
    - Learned neural network models (e.g., GRU-based predictors)
    - Hybrid models combining learned and analytical components
    """

    @abstractmethod
    def step(
        self,
        model: mjx.Model,
        data: mjx.Data,
        state_history: jax.Array | None = None,
    ) -> mjx.Data:
        """Advance the state by one timestep using custom dynamics.

        This method should compute the next state given the current state
        and control input. The control input is available in data.ctrl.

        Args:
            model: The MJX model (for reference - dimensions, limits, etc.)
            data: The current state, including data.qpos, data.qvel, data.ctrl
            state_history: Optional history of (state, action) pairs for
                          models that require temporal context (e.g., RNNs).
                          Shape: (history_length, state_action_dim)

        Returns:
            The next state as an mjx.Data object with updated qpos, qvel,
            and any other relevant fields.
        """
        pass

    def initialize_state_history(
        self, model: mjx.Model, data: mjx.Data | None = None
    ) -> jax.Array:
        """Initialize the state-action history buffer.

        This is used for models that require temporal context (e.g., GRU).
        The default implementation creates a zero-filled history buffer.

        Args:
            model: The MJX model
            data: The current state (optional, not used in default implementation)

        Returns:
            Initialized history buffer. For models that don't need history,
            returns None. For models that do (e.g., GRU), returns array of
            shape (history_length, state_action_dim)
        """
        # Default implementation for models that don't need history
        return None

    def update_state_history(
        self,
        state_history: jax.Array,
        data: mjx.Data,
        control: jax.Array,
    ) -> jax.Array:
        """Update the state-action history with the current step.

        Implements a sliding window: drops oldest entry and appends current.

        Args:
            state_history: Current history buffer, shape (history_length, dim)
            data: Current state
            control: Current control action

        Returns:
            Updated history buffer with the same shape
        """
        # Concatenate current state and action
        state_action = jnp.concatenate([data.qpos, data.qvel, control])
        # Shift history and append new entry
        updated_history = jnp.roll(state_history, shift=-1, axis=0)
        updated_history = updated_history.at[-1].set(state_action)
        return updated_history

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for this dynamics model.

        Useful for logging, saving, and reproducing experiments.
        """
        return {
            "type": self.__class__.__name__,
        }
