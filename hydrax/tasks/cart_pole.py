import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from typing import TYPE_CHECKING

from hydrax import ROOT
from hydrax.task_base import Task

if TYPE_CHECKING:
    from hydrax.dynamics_base import DynamicsModel


class CartPole(Task):
    """A cart-pole swingup task."""

    def __init__(self, custom_dynamics: "DynamicsModel | None" = None) -> None:
        """Load the MuJoCo model and set task parameters.

        Args:
            custom_dynamics: Optional custom dynamics model to use for rollouts
                           instead of the default MJX physics simulator.
        """
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/cart_pole/scene.xml"
        )
        super().__init__(mj_model, trace_sites=["tip"], custom_dynamics=custom_dynamics)

    def _distance_to_upright(self, state: mjx.Data) -> jax.Array:
        """Get a measure of distance to the upright position."""
        theta = state.qpos[1] + jnp.pi
        theta_err = jnp.array([jnp.cos(theta) - 1, jnp.sin(theta)])
        return jnp.sum(jnp.square(theta_err))

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ)."""
        theta_cost = self._distance_to_upright(state)
        centering_cost = jnp.sum(jnp.square(state.qpos[0]))
        velocity_cost = 0.01 * jnp.sum(jnp.square(state.qvel))
        control_cost = 0.01 * jnp.sum(jnp.square(control))
        return theta_cost + centering_cost + velocity_cost + control_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        theta_cost = 10 * self._distance_to_upright(state)
        centering_cost = jnp.sum(jnp.square(state.qpos[0]))
        velocity_cost = 0.01 * jnp.sum(jnp.square(state.qvel))
        return theta_cost + centering_cost + velocity_cost
