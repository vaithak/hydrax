"""Data loader for training dynamics models from logged CSV data.

This module provides utilities to load state-action trajectories from CSV files
logged during MuJoCo/MJX simulations and prepare them for training neural network
dynamics models.
"""

from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


def load_csv_data(csv_path: str) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Load a single CSV file containing logged state-action data.

    Expected CSV format:
        timestamp, qpos_0, qpos_1, ..., qvel_0, qvel_1, ..., ctrl_0, ctrl_1, ...

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (qpos, qvel, ctrl) arrays:
            - qpos: Array of shape (T, nq) containing positions
            - qvel: Array of shape (T, nv) containing velocities
            - ctrl: Array of shape (T, nu) containing control inputs
    """
    df = pd.read_csv(csv_path)

    # Separate columns by type
    qpos_cols = [col for col in df.columns if col.startswith('qpos_')]
    qvel_cols = [col for col in df.columns if col.startswith('qvel_')]
    ctrl_cols = [col for col in df.columns if col.startswith('ctrl_')]

    # Extract data
    qpos = df[qpos_cols].values
    qvel = df[qvel_cols].values
    ctrl = df[ctrl_cols].values

    return jnp.array(qpos), jnp.array(qvel), jnp.array(ctrl)


class DynamicsDataset:
    """Dataset for training dynamics models with windowed state-action pairs.

    This dataset creates training examples consisting of:
        - Input: History of (state, action) pairs over a sliding window (normalized)
        - Target: Sequence of delta states over prediction horizon

    The dataset handles:
        - Angle transformation (angle -> cos/sin)
        - Coordinate normalization relative to first point in history
        - Multi-step autoregressive prediction targets
    """

    def __init__(
        self,
        data_dir: str,
        history_length: int = 11,
        prediction_horizon: int = 9,
        coordinate_indices: jnp.ndarray | None = None,
        angle_indices: jnp.ndarray | None = None,
    ):
        """Initialize the dynamics dataset.

        Args:
            data_dir: Directory containing CSV files with logged trajectories
            history_length: Number of past timesteps to use as input
            prediction_horizon: Number of steps ahead to predict autoregressively
            coordinate_indices: Indices in qpos for Cartesian coordinates (for normalization)
            angle_indices: Indices in qpos that are angles (for cos/sin transformation)
        """
        self.data_dir = Path(data_dir)
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.coordinate_indices = coordinate_indices if coordinate_indices is not None else jnp.array([])
        self.angle_indices = angle_indices if angle_indices is not None else jnp.array([])

        # Pre-sort angle indices for consistent processing
        if len(self.angle_indices) > 0:
            self.sorted_angle_indices = [int(idx) for idx in jnp.sort(self.angle_indices)]
        else:
            self.sorted_angle_indices = []

        # Load all CSV files
        self.trajectories = self._load_all_trajectories()

        # Compute dimensions after transformation
        if len(self.trajectories) > 0:
            sample_qpos = self.trajectories[0][0][0]  # First qpos from first trajectory
            transformed_qpos = self._transform_qpos(sample_qpos)
            self.transformed_qpos_dim = len(transformed_qpos)
            self.nv = self.trajectories[0][1].shape[1]  # qvel dimension

            # Compute transformed coordinate indices
            self.transformed_coord_indices = self._compute_transformed_coord_indices()
        else:
            raise ValueError("No trajectories found in data directory")

        # Create windowed dataset
        self.examples = self._create_examples()

        print(f"Loaded {len(self.trajectories)} trajectories")
        print(f"Created {len(self.examples)} training examples")
        print(f"Original qpos dim: {len(sample_qpos)}, Transformed qpos dim: {self.transformed_qpos_dim}")

    def _load_all_trajectories(self) -> List[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Load all CSV files from the data directory."""
        trajectories = []
        csv_files = sorted(self.data_dir.glob("*.csv"))

        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")

        for csv_file in csv_files:
            qpos, qvel, ctrl = load_csv_data(str(csv_file))
            trajectories.append((qpos, qvel, ctrl))

        return trajectories

    def _compute_transformed_coord_indices(self) -> List[int]:
        """Compute coordinate indices after angle expansion."""
        if len(self.coordinate_indices) == 0:
            return []

        transformed_indices = []
        for coord_idx in self.coordinate_indices:
            # Each angle before this index adds one extra dimension
            num_angles_before = sum(1 for angle_idx in self.sorted_angle_indices if angle_idx < coord_idx)
            transformed_idx = int(coord_idx) + num_angles_before
            transformed_indices.append(transformed_idx)

        return transformed_indices

    def _transform_qpos(self, qpos: jnp.ndarray) -> jnp.ndarray:
        """Transform qpos (convert angles to cos/sin).

        Args:
            qpos: Original qpos array, shape (..., nq)

        Returns:
            Transformed qpos with angles expanded to (cos, sin), shape (..., transformed_qpos_dim)
        """
        if len(self.sorted_angle_indices) == 0:
            return qpos

        # Build transformed array by processing segments between angles
        parts = []
        prev_idx = 0

        for angle_idx in self.sorted_angle_indices:
            # Add regular values before this angle
            if angle_idx > prev_idx:
                parts.append(qpos[..., prev_idx:angle_idx])
            # Add cos/sin for the angle
            angle_val = qpos[..., angle_idx]
            parts.append(jnp.expand_dims(jnp.cos(angle_val), -1))
            parts.append(jnp.expand_dims(jnp.sin(angle_val), -1))
            prev_idx = angle_idx + 1

        # Add any remaining regular values
        if prev_idx < qpos.shape[-1]:
            parts.append(qpos[..., prev_idx:])

        return jnp.concatenate(parts, axis=-1)

    def _normalize_coordinates(
        self,
        transformed_state_action_hist: jnp.ndarray
    ) -> jnp.ndarray:
        """Normalize coordinates relative to first timestep in history.

        Args:
            transformed_state_action_hist: History with transformed qpos,
                                          shape (history_length, state_dim + action_dim)

        Returns:
            History with normalized coordinates
        """
        if len(self.transformed_coord_indices) == 0:
            return transformed_state_action_hist

        # Extract first frame's coordinate values as reference
        reference_qpos = transformed_state_action_hist[0, :self.transformed_qpos_dim]
        normalized = transformed_state_action_hist.copy()

        # Subtract reference coordinates from all frames
        for idx in self.transformed_coord_indices:
            normalized = normalized.at[:, idx].add(-reference_qpos[idx])

        return normalized

    def _compute_delta_qpos(
        self,
        current_qpos_transformed: jnp.ndarray,
        next_qpos_transformed: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute delta for qpos in transformed space.

        For angles (represented as cos/sin pairs), the delta is computed as:
            delta_cos = cos(next - current)
            delta_sin = sin(next - current)

        Args:
            current_qpos_transformed: Current qpos in transformed space
            next_qpos_transformed: Next qpos in transformed space

        Returns:
            Delta in transformed space
        """
        if len(self.sorted_angle_indices) == 0:
            # No angles, simple subtraction
            return next_qpos_transformed - current_qpos_transformed

        # Build delta by processing each segment
        delta_parts = []
        transformed_idx = 0
        original_idx = 0

        for angle_idx in self.sorted_angle_indices:
            # Delta for regular positions before this angle
            segment_length = angle_idx - original_idx
            if segment_length > 0:
                delta_parts.append(
                    next_qpos_transformed[transformed_idx:transformed_idx + segment_length] -
                    current_qpos_transformed[transformed_idx:transformed_idx + segment_length]
                )
                transformed_idx += segment_length

            # Delta for angle (compute cos(delta), sin(delta))
            # We have cos(current), sin(current) and cos(next), sin(next)
            cos_current = current_qpos_transformed[transformed_idx]
            sin_current = current_qpos_transformed[transformed_idx + 1]
            cos_next = next_qpos_transformed[transformed_idx]
            sin_next = next_qpos_transformed[transformed_idx + 1]

            # Using angle difference formula:
            # cos(next - current) = cos(next)*cos(current) + sin(next)*sin(current)
            # sin(next - current) = sin(next)*cos(current) - cos(next)*sin(current)
            cos_delta = cos_next * cos_current + sin_next * sin_current
            sin_delta = sin_next * cos_current - cos_next * sin_current

            delta_parts.append(jnp.array([cos_delta, sin_delta]))

            transformed_idx += 2
            original_idx = angle_idx + 1

        # Delta for remaining positions after last angle
        if transformed_idx < len(current_qpos_transformed):
            delta_parts.append(
                next_qpos_transformed[transformed_idx:] -
                current_qpos_transformed[transformed_idx:]
            )

        return jnp.concatenate(delta_parts)

    def _create_examples(self) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Create windowed training examples from trajectories.

        Each example consists of:
            - state_action_history: (history_length, state_dim + action_dim) - normalized
            - targets: (prediction_horizon, state_dim) - deltas in transformed space
        """
        examples = []

        for qpos_traj, qvel_traj, ctrl_traj in self.trajectories:
            T = len(qpos_traj)

            # Create overlapping windows
            for t in range(self.history_length, T - self.prediction_horizon + 1):
                # Get history window [t-history_length : t]
                qpos_hist = qpos_traj[t - self.history_length:t]
                qvel_hist = qvel_traj[t - self.history_length:t]
                ctrl_hist = ctrl_traj[t - self.history_length:t]

                # Transform qpos in history
                qpos_hist_transformed = self._transform_qpos(qpos_hist)

                # Combine state and action for history
                state_hist = jnp.concatenate([qpos_hist_transformed, qvel_hist], axis=-1)
                state_action_hist = jnp.concatenate([state_hist, ctrl_hist], axis=-1)

                # Normalize coordinates relative to first point in history
                state_action_hist_normalized = self._normalize_coordinates(state_action_hist)

                # Get targets: deltas for prediction_horizon steps
                target_deltas = []

                for h in range(self.prediction_horizon):
                    # Current state at t + h - 1 (or t-1 for h=0)
                    current_idx = t + h - 1
                    next_idx = t + h

                    current_qpos = qpos_traj[current_idx]
                    next_qpos = qpos_traj[next_idx]
                    current_qvel = qvel_traj[current_idx]
                    next_qvel = qvel_traj[next_idx]

                    # Transform positions
                    current_qpos_transformed = self._transform_qpos(current_qpos)
                    next_qpos_transformed = self._transform_qpos(next_qpos)

                    # Compute delta qpos (handling angles properly)
                    delta_qpos = self._compute_delta_qpos(
                        current_qpos_transformed,
                        next_qpos_transformed
                    )

                    # Delta qvel is simple subtraction
                    delta_qvel = next_qvel - current_qvel

                    # Combine into full state delta
                    delta_state = jnp.concatenate([delta_qpos, delta_qvel])
                    target_deltas.append(delta_state)

                # Stack targets
                targets = jnp.stack(target_deltas)  # (prediction_horizon, state_dim)

                examples.append((state_action_hist_normalized, targets))

        return examples

    def __len__(self) -> int:
        """Return the number of training examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get a training example by index.

        Returns:
            Tuple of (state_action_history, targets)
                - state_action_history: (history_length, state_dim + action_dim)
                - targets: (prediction_horizon, state_dim)
        """
        return self.examples[idx]

    def get_batch(self, indices: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get a batch of training examples.

        Args:
            indices: Array of indices to retrieve

        Returns:
            Tuple of batched (state_action_history, targets)
                - state_action_history: (batch_size, history_length, state_dim + action_dim)
                - targets: (batch_size, prediction_horizon, state_dim)
        """
        batch_state_action = []
        batch_targets = []

        for idx in indices:
            state_action, targets = self.examples[int(idx)]
            batch_state_action.append(state_action)
            batch_targets.append(targets)

        return (
            jnp.stack(batch_state_action),
            jnp.stack(batch_targets),
        )

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple["DynamicsDataset", "DynamicsDataset"]:
        """Split dataset into train and validation sets.

        Args:
            train_ratio: Fraction of data to use for training
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        n_train = int(len(self.examples) * train_ratio)

        # Shuffle indices
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self.examples))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Create new datasets with subsets
        train_dataset = DynamicsDataset.__new__(DynamicsDataset)
        train_dataset.data_dir = self.data_dir
        train_dataset.history_length = self.history_length
        train_dataset.prediction_horizon = self.prediction_horizon
        train_dataset.coordinate_indices = self.coordinate_indices
        train_dataset.angle_indices = self.angle_indices
        train_dataset.sorted_angle_indices = self.sorted_angle_indices
        train_dataset.transformed_qpos_dim = self.transformed_qpos_dim
        train_dataset.nv = self.nv
        train_dataset.transformed_coord_indices = self.transformed_coord_indices
        train_dataset.trajectories = self.trajectories
        train_dataset.examples = [self.examples[i] for i in train_indices]

        val_dataset = DynamicsDataset.__new__(DynamicsDataset)
        val_dataset.data_dir = self.data_dir
        val_dataset.history_length = self.history_length
        val_dataset.prediction_horizon = self.prediction_horizon
        val_dataset.coordinate_indices = self.coordinate_indices
        val_dataset.angle_indices = self.angle_indices
        val_dataset.sorted_angle_indices = self.sorted_angle_indices
        val_dataset.transformed_qpos_dim = self.transformed_qpos_dim
        val_dataset.nv = self.nv
        val_dataset.transformed_coord_indices = self.transformed_coord_indices
        val_dataset.trajectories = self.trajectories
        val_dataset.examples = [self.examples[i] for i in val_indices]

        return train_dataset, val_dataset
