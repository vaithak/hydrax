"""Custom dynamics models for rollout predictions.

This package provides alternative dynamics models that can be used
in place of the default MJX physics simulator for controller rollouts.
"""

from hydrax.dynamics.deterministic_dynamics import (
    DoubleCartPoleDynamics,
    SimplifiedPhysicsDynamics,
)
from hydrax.dynamics.nn_dynamics import NeuralNetworkDynamics

__all__ = [
    "NeuralNetworkDynamics",
    "SimplifiedPhysicsDynamics",
    "DoubleCartPoleDynamics",
]
