"""Deterministic simplified physics dynamics model.

This module implements a simplified analytical physics model for systems
where the full MuJoCo simulation might be too complex or computationally
expensive for planning.
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
from mujoco import mjx

from hydrax.dynamics_base import DynamicsModel


class SimplifiedPhysicsDynamics(DynamicsModel):
    """Simplified analytical physics dynamics model.

    This is a template for implementing custom deterministic physics models.
    The default implementation uses basic Euler integration for simple systems.

    For specific systems (e.g., cart-pole, pendulum), you should subclass this
    and override the compute_acceleration method to implement the actual physics.
    """

    def __init__(
        self,
        model: mjx.Model,
        integration_method: str = "euler",
        use_damping: bool = True,
    ):
        """Initialize the simplified physics model.

        Args:
            model: The MJX model (for reference dimensions and parameters)
            integration_method: Integration method ("euler", "rk4")
            use_damping: Whether to include velocity damping
        """
        self.model_ref = model
        self.integration_method = integration_method
        self.use_damping = use_damping
        self.dt = model.opt.timestep

    def compute_acceleration(
        self, model: mjx.Model, qpos: jax.Array, qvel: jax.Array, ctrl: jax.Array
    ) -> jax.Array:
        """Compute accelerations from current state and control.

        This is the core physics computation that should be overridden
        for specific systems. The default implementation uses a very
        simple linear approximation.

        Args:
            model: The MJX model
            qpos: Generalized positions
            qvel: Generalized velocities
            ctrl: Control inputs

        Returns:
            qacc: Generalized accelerations
        """
        # Simple linear approximation (override this for real systems!)
        # This is just a placeholder - you should implement actual physics here

        # Extract mass matrix diagonal (simplified)
        # In a real implementation, you'd compute M^{-1} * (tau - C - G)
        nv = model.nv

        # Simple damping
        damping = -0.1 * qvel if self.use_damping else jnp.zeros_like(qvel)

        # Control forces (assuming direct force control)
        # Map controls to generalized forces based on actuator configuration
        ctrl_forces = jnp.zeros(nv)
        if model.nu > 0:
            # Simple 1-to-1 mapping (should be customized per system)
            ctrl_forces = ctrl_forces.at[: model.nu].set(ctrl)

        # Simple acceleration (override with actual dynamics!)
        qacc = ctrl_forces + damping

        return qacc

    def step(
        self,
        model: mjx.Model,
        data: mjx.Data,
        state_history: jax.Array | None = None,
    ) -> mjx.Data:
        """Advance the state using simplified physics.

        Args:
            model: The MJX model
            data: Current state
            state_history: Not used for deterministic physics

        Returns:
            Updated state
        """
        qpos = data.qpos
        qvel = data.qvel
        ctrl = data.ctrl

        if self.integration_method == "euler":
            # Forward Euler integration
            qacc = self.compute_acceleration(model, qpos, qvel, ctrl)
            new_qvel = qvel + self.dt * qacc
            new_qpos = qpos + self.dt * new_qvel

        elif self.integration_method == "rk4":
            # Runge-Kutta 4th order integration
            def dynamics(q, v):
                a = self.compute_acceleration(model, q, v, ctrl)
                return v, a

            # RK4 for second-order system
            k1_v, k1_a = dynamics(qpos, qvel)
            k2_v, k2_a = dynamics(qpos + 0.5 * self.dt * k1_v, qvel + 0.5 * self.dt * k1_a)
            k3_v, k3_a = dynamics(qpos + 0.5 * self.dt * k2_v, qvel + 0.5 * self.dt * k2_a)
            k4_v, k4_a = dynamics(qpos + self.dt * k3_v, qvel + self.dt * k3_a)

            new_qpos = qpos + (self.dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
            new_qvel = qvel + (self.dt / 6.0) * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)

        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")

        # Update data
        next_data = data.replace(
            qpos=new_qpos,
            qvel=new_qvel,
            time=data.time + self.dt,
        )

        return next_data

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            "type": "SimplifiedPhysicsDynamics",
            "integration_method": self.integration_method,
            "use_damping": self.use_damping,
            "dt": self.dt,
        }


class DoubleCartPoleDynamics(SimplifiedPhysicsDynamics):
    """Simplified dynamics for double cart-pole system.

    Implements the analytical equations of motion for a cart with
    two pendulums, using a simplified model (e.g., massless rods).
    """

    def __init__(
        self,
        model: mjx.Model,
        integration_method: str = "rk4",
        cart_mass: float = 1.0,
        pole1_mass: float = 0.1,
        pole2_mass: float = 0.1,
        pole1_length: float = 0.5,
        pole2_length: float = 0.5,
        gravity: float = 9.81,
    ):
        """Initialize double cart-pole dynamics.

        Args:
            model: The MJX model
            integration_method: Integration method
            cart_mass: Mass of the cart
            pole1_mass: Mass of first pole
            pole2_mass: Mass of second pole
            pole1_length: Length of first pole
            pole2_length: Length of second pole
            gravity: Gravitational acceleration
        """
        super().__init__(model, integration_method, use_damping=True)
        self.mc = cart_mass
        self.m1 = pole1_mass
        self.m2 = pole2_mass
        self.l1 = pole1_length
        self.l2 = pole2_length
        self.g = gravity

    def compute_acceleration(
        self, model: mjx.Model, qpos: jax.Array, qvel: jax.Array, ctrl: jax.Array
    ) -> jax.Array:
        """Compute accelerations for double cart-pole.

        State: [x, theta1, theta2] (cart position, pole angles)
        Control: [F] (force on cart)

        This implements simplified equations of motion.
        """
        # Extract state components
        x, theta1, theta2 = qpos[0], qpos[1], qpos[2]
        x_dot, theta1_dot, theta2_dot = qvel[0], qvel[1], qvel[2]

        # Extract control
        F = ctrl[0] if len(ctrl) > 0 else 0.0

        # Simplified equations (linearized around upright for demonstration)
        # For a full implementation, use the complete nonlinear equations

        # Mass matrix and forcing terms (simplified)
        # This is a placeholder - you would implement the full dynamics here
        # using the Lagrangian or Newton-Euler formulation

        # Simple approximation for demonstration
        x_acc = (F - 0.1 * x_dot) / (self.mc + self.m1 + self.m2)
        theta1_acc = (self.g / self.l1) * jnp.sin(theta1) - 0.1 * theta1_dot
        theta2_acc = (self.g / self.l2) * jnp.sin(theta2) - 0.1 * theta2_dot

        return jnp.array([x_acc, theta1_acc, theta2_acc])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        config = super().get_config()
        config.update(
            {
                "type": "DoubleCartPoleDynamics",
                "cart_mass": self.mc,
                "pole1_mass": self.m1,
                "pole2_mass": self.m2,
                "pole1_length": self.l1,
                "pole2_length": self.l2,
                "gravity": self.g,
            }
        )
        return config
