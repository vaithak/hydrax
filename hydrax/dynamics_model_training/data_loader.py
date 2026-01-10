"""Data loader for training dynamics models from logged CSV data.

This module provides utilities to load state-action trajectories from CSV files
logged during MuJoCo/MJX simulations and prepare them for training neural network
dynamics models.
"""

from pathlib import Path
from typing import List, Tuple

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

    This dataset creates training examples consisting of single tensors of size
    (history_length + prediction_horizon, transformed_state_control_dim).

    The dataset handles:
        - Angle transformation (angle -> cos/sin)
        - Velocity normalization to [-1, 1] range
        - Creating sliding windows of state-control sequences

    Note: Coordinate normalization and delta computation are handled in the training script.
    """

    def __init__(
        self,
        data_dir: str,
        history_length: int = 11,
        prediction_horizon: int = 9,
        coordinate_indices: jnp.ndarray | None = None,
        angle_indices: jnp.ndarray | None = None,
        normalize_velocities: bool = True,
    ):
        """Initialize the dynamics dataset.

        Args:
            data_dir: Directory containing CSV files with logged trajectories
            history_length: Number of past timesteps to use as input
            prediction_horizon: Number of steps ahead to predict autoregressively
            coordinate_indices: Indices in qpos for Cartesian coordinates (for normalization in training)
            angle_indices: Indices in qpos that are angles (for cos/sin transformation)
            normalize_velocities: If True, normalize velocities to [-1, 1] range
        """
        self.data_dir = Path(data_dir)
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.coordinate_indices = coordinate_indices if coordinate_indices is not None else jnp.array([])
        self.angle_indices = angle_indices if angle_indices is not None else jnp.array([])
        self.normalize_velocities = normalize_velocities

        # Pre-sort angle indices for consistent processing
        if len(self.angle_indices) > 0:
            self.sorted_angle_indices = [int(idx) for idx in jnp.sort(self.angle_indices)]
        else:
            self.sorted_angle_indices = []

        # Load all CSV files
        self.trajectories = self._load_all_trajectories()

        # Compute velocity normalization statistics
        if self.normalize_velocities:
            self.qvel_min, self.qvel_max = self._compute_velocity_stats()
            # Add small epsilon to avoid division by zero
            self.qvel_range = self.qvel_max - self.qvel_min
            self.qvel_range = jnp.maximum(self.qvel_range, 1e-6)
        else:
            self.qvel_min = None
            self.qvel_max = None
            self.qvel_range = None

        # Compute dimensions after transformation
        if len(self.trajectories) > 0:
            sample_qpos = self.trajectories[0][0][0]  # First qpos from first trajectory
            transformed_qpos = self._transform_qpos(sample_qpos)
            self.transformed_qpos_dim = len(transformed_qpos)
            self.nv = self.trajectories[0][1].shape[1]  # qvel dimension
            self.nu = self.trajectories[0][2].shape[1]  # control dimension

            # Total dimension: transformed_qpos + qvel + ctrl
            self.transformed_state_control_dim = self.transformed_qpos_dim + self.nv + self.nu

            # Compute transformed coordinate indices (for use in training script)
            self.transformed_coord_indices = self._compute_transformed_coord_indices()
        else:
            raise ValueError("No trajectories found in data directory")

        # Create windowed dataset
        self.examples = self._create_examples()

        print(f"Loaded {len(self.trajectories)} trajectories")
        print(f"Created {len(self.examples)} training examples")
        print(f"Original qpos dim: {len(sample_qpos)}, Transformed qpos dim: {self.transformed_qpos_dim}")
        print(f"Transformed state+control dim: {self.transformed_state_control_dim}")
        print(f"Example shape: ({self.history_length + self.prediction_horizon}, {self.transformed_state_control_dim})")
        if self.normalize_velocities:
            print(f"Velocity normalization enabled:")
            print(f"  qvel_min: {self.qvel_min}")
            print(f"  qvel_max: {self.qvel_max}")

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

    def _compute_velocity_stats(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute min and max velocity values across all trajectories.

        Returns:
            Tuple of (qvel_min, qvel_max) arrays of shape (nv,)
        """
        all_qvel = []
        for qpos, qvel, ctrl in self.trajectories:
            all_qvel.append(qvel)

        # Concatenate all velocities
        all_qvel = jnp.concatenate(all_qvel, axis=0)

        # Compute min and max for each velocity dimension
        qvel_min = jnp.min(all_qvel, axis=0)
        qvel_max = jnp.max(all_qvel, axis=0)

        return qvel_min, qvel_max

    def _normalize_qvel(self, qvel: jnp.ndarray) -> jnp.ndarray:
        """Normalize velocities to [-1, 1] range.

        Args:
            qvel: Velocity array of shape (..., nv)

        Returns:
            Normalized velocity array
        """
        if not self.normalize_velocities:
            return qvel

        # Scale to [-1, 1]: (qvel - min) / (max - min) * 2 - 1
        normalized = (qvel - self.qvel_min) / self.qvel_range * 2.0 - 1.0
        return normalized

    def denormalize_qvel(self, qvel_normalized: jnp.ndarray) -> jnp.ndarray:
        """Denormalize velocities from [-1, 1] range back to original scale.

        Args:
            qvel_normalized: Normalized velocity array of shape (..., nv)

        Returns:
            Denormalized velocity array
        """
        if not self.normalize_velocities:
            return qvel_normalized

        # Inverse scaling: (qvel_norm + 1) / 2 * (max - min) + min
        denormalized = (qvel_normalized + 1.0) / 2.0 * self.qvel_range + self.qvel_min
        return denormalized

    def _compute_transformed_coord_indices(self) -> List[int]:
        """Compute coordinate indices after angle expansion.

        This is used by the training script for normalization.
        """
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

    def _create_examples(self) -> List[jnp.ndarray]:
        """Create windowed training examples from trajectories.

        Each example is a single tensor of shape:
            (history_length + prediction_horizon, transformed_state_control_dim)

        The tensor contains:
            - First history_length timesteps: state + control at [t-history_length:t]
            - Next prediction_horizon timesteps: state + control at [t:t+prediction_horizon]

        Note: The last control input in each example won't be used in training,
        but is included for simplicity. The training script will handle extracting
        the relevant slices for history and prediction.
        """
        examples = []

        for qpos_traj, qvel_traj, ctrl_traj in self.trajectories:
            T = len(qpos_traj)

            # Create overlapping windows
            # We need history_length + prediction_horizon timesteps total
            window_size = self.history_length + self.prediction_horizon

            for t in range(T - window_size + 1):
                # Get window [t : t + window_size]
                qpos_window = qpos_traj[t:t + window_size]
                qvel_window = qvel_traj[t:t + window_size]
                ctrl_window = ctrl_traj[t:t + window_size]

                # Transform qpos for all timesteps in window
                qpos_window_transformed = self._transform_qpos(qpos_window)

                # Normalize velocities
                qvel_window_normalized = self._normalize_qvel(qvel_window)

                # Combine state and control: [transformed_qpos, normalized_qvel, ctrl]
                state_control_window = jnp.concatenate(
                    [qpos_window_transformed, qvel_window_normalized, ctrl_window],
                    axis=-1
                )

                examples.append(state_control_window)

        return examples

    def __len__(self) -> int:
        """Return the number of training examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> jnp.ndarray:
        """Get a training example by index.

        Returns:
            Single tensor of shape (history_length + prediction_horizon, transformed_state_control_dim)
            containing the complete sequence (history + future states)
        """
        return self.examples[idx]

    def get_batch(self, indices: jnp.ndarray) -> jnp.ndarray:
        """Get a batch of training examples.

        Args:
            indices: Array of indices to retrieve

        Returns:
            Batched examples of shape (batch_size, history_length + prediction_horizon, transformed_state_control_dim)
            containing complete sequences (history + future states)
        """
        batch_examples = []

        for idx in indices:
            batch_examples.append(self.examples[int(idx)])

        return jnp.stack(batch_examples)

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
        train_dataset.normalize_velocities = self.normalize_velocities
        train_dataset.qvel_min = self.qvel_min
        train_dataset.qvel_max = self.qvel_max
        train_dataset.qvel_range = self.qvel_range
        train_dataset.transformed_qpos_dim = self.transformed_qpos_dim
        train_dataset.nv = self.nv
        train_dataset.nu = self.nu
        train_dataset.transformed_state_control_dim = self.transformed_state_control_dim
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
        val_dataset.normalize_velocities = self.normalize_velocities
        val_dataset.qvel_min = self.qvel_min
        val_dataset.qvel_max = self.qvel_max
        val_dataset.qvel_range = self.qvel_range
        val_dataset.transformed_qpos_dim = self.transformed_qpos_dim
        val_dataset.nv = self.nv
        val_dataset.nu = self.nu
        val_dataset.transformed_state_control_dim = self.transformed_state_control_dim
        val_dataset.transformed_coord_indices = self.transformed_coord_indices
        val_dataset.trajectories = self.trajectories
        val_dataset.examples = [self.examples[i] for i in val_indices]

        return train_dataset, val_dataset
