# Neural Network Dynamics Model Training

This module provides tools for training GRU-based neural network dynamics models on logged state-action trajectory data from MuJoCo/MJX simulations.

## Overview

The training system supports:
- **Multi-step autoregressive prediction**: Predicts multiple steps ahead by rolling out predictions
- **Angle transformation**: Converts angles to (cos, sin) representation for better learning
- **Coordinate normalization**: Normalizes Cartesian coordinates relative to the first point in history
- **Delta prediction**: Network predicts state changes (deltas) rather than absolute states

## Directory Structure

```
dynamics_model_training/
├── __init__.py              # Module initialization
├── data_loader.py           # Dataset and data loading utilities
├── train.py                 # Training loop and loss functions
├── evaluate.py              # Evaluation and visualization tools
├── train_double_cart_pole.py  # Example training script
├── checkpoints/             # Saved model checkpoints
│   └── <task_name>/
│       ├── best_model.pkl
│       └── model_epoch_*.pkl
└── README.md                # This file
```

## Quick Start

### 1. Train a Model

For the double cart pole task:

```bash
cd /home/ubuntu/LeggedRobots/hydrax
python -m hydrax.dynamics_model_training.train_double_cart_pole
```

### 2. Evaluate a Trained Model

```bash
python -m hydrax.dynamics_model_training.evaluate \
    --checkpoint hydrax/dynamics_model_training/checkpoints/double_cart_pole/best_model.pkl \
    --data-dir hydrax/data_csv/double_cart_pole \
    --output-dir results/evaluation \
    --num-examples 100
```

## Training Your Own Model

### Step 1: Prepare Data

Ensure your CSV data is in the correct format:
```
timestamp,qpos_0,qpos_1,...,qvel_0,qvel_1,...,ctrl_0,ctrl_1,...
0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
0.01,0.002,0.003,-0.001,0.2,0.3,-0.1,0.9
...
```

### Step 2: Create Training Script

```python
import jax.numpy as jnp
from hydrax.dynamics_model_training.train import train_dynamics_model

# Define task-specific indices
coordinate_indices = jnp.array([0])  # Indices of Cartesian coordinates in qpos
angle_indices = jnp.array([1, 2])    # Indices of angles in qpos

# Train the model
network, history = train_dynamics_model(
    data_dir='path/to/csv/data',
    task_name='my_task',
    coordinate_indices=coordinate_indices,
    angle_indices=angle_indices,
    hidden_size=128,
    history_length=11,
    prediction_horizon=9,
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-5,
    seed=42,
)
```

### Step 3: Load and Use Trained Model

```python
from hydrax.dynamics.nn_dynamics import NeuralNetworkDynamics
import mujoco
from mujoco import mjx

# Load MuJoCo model
mj_model = mujoco.MjModel.from_xml_path('path/to/model.xml')
model = mjx.put_model(mj_model)

# Load trained dynamics model
dynamics = NeuralNetworkDynamics.load_params(
    model=model,
    path='hydrax/dynamics_model_training/checkpoints/my_task/best_model.pkl'
)

# Use in task
from hydrax.tasks.my_task import MyTask
task = MyTask(custom_dynamics=dynamics)
```

## Configuration Parameters

### Data Parameters
- `data_dir`: Directory containing CSV trajectory files
- `coordinate_indices`: Indices in qpos that are Cartesian coordinates (normalized)
- `angle_indices`: Indices in qpos that are angles (transformed to cos/sin)

### Model Parameters
- `hidden_size`: Size of GRU hidden state (default: 128)
- `history_length`: Number of past timesteps to condition on (default: 11)
- `prediction_horizon`: Number of steps to predict autoregressively (default: 9)

### Training Parameters
- `batch_size`: Training batch size (default: 32)
- `num_epochs`: Number of training epochs (default: 100)
- `learning_rate`: Learning rate for AdamW optimizer (default: 1e-3)
- `weight_decay`: Weight decay for regularization (default: 1e-5)
- `seed`: Random seed for reproducibility

## Model Architecture

The dynamics model uses a GRU (Gated Recurrent Unit) architecture:

1. **Input**: History of (state, action) pairs
   - States are transformed (angles → cos/sin, coordinates normalized)
   - Shape: `(history_length, state_dim + action_dim)`

2. **GRU Processing**: Processes temporal sequence
   - Hidden size: configurable (default 128)
   - Captures temporal dependencies

3. **Output Layer**: Predicts state delta
   - Shape: `(state_dim,)` where state = transformed_qpos + qvel
   - Delta is applied using:
     - Angles: Trigonometric addition formulas
     - Other states: Simple addition

## Loss Function

The model is trained with autoregressive multi-step prediction:

```
For each step h in prediction_horizon:
    1. Predict delta_h from current history
    2. Compute MSE loss: ||predicted_delta_h - target_delta_h||²
    3. Apply delta_h to current state
    4. Update history with new state-action pair
    5. Repeat for next step

Total loss = mean over all steps
```

This approach ensures the model learns to predict correctly even when using its own predictions.

## Checkpoints

Models are saved at:
- `checkpoints/<task_name>/best_model.pkl`: Best validation loss
- `checkpoints/<task_name>/model_epoch_<N>.pkl`: Every 10 epochs

Checkpoint contents:
- Network parameters (graph_def and state)
- Configuration (dimensions, indices, hyperparameters)
- Training history (train/val losses)

## Evaluation

The evaluation script computes:
- MSE and MAE for each prediction step
- Averaged over validation examples
- Plots showing error growth over prediction horizon

Example output:
```
Average MSE per step:
  Step 1: 0.000123
  Step 2: 0.000245
  ...
  Step 9: 0.001234
```

## Tips for Better Performance

1. **Data Quality**: Ensure diverse trajectories covering the state space
2. **Normalization**: Properly set coordinate_indices and angle_indices
3. **History Length**: Longer history helps but increases computation
4. **Prediction Horizon**: Balance between short-term accuracy and long-term planning
5. **Hidden Size**: Increase for complex dynamics, decrease for simpler systems
6. **Learning Rate**: Use learning rate scheduling for better convergence
7. **Regularization**: Adjust weight_decay to prevent overfitting

## Troubleshooting

### High Training Loss
- Increase hidden_size
- Decrease learning_rate
- Check data normalization
- Verify angle_indices are correct

### Overfitting (low train, high val loss)
- Increase weight_decay
- Reduce hidden_size
- Add more training data
- Use dropout (modify network architecture)

### NaN Loss
- Decrease learning_rate
- Check for invalid data (NaN, Inf)
- Verify angle transformations are correct
- Reduce prediction_horizon initially

## Advanced Usage

### Custom State Transformations

You can implement custom transformations in the dataset:

```python
def custom_transform(qpos):
    # Custom transformation logic
    return transformed_qpos

dataset = DynamicsDataset(
    data_dir='...',
    transform_fn=custom_transform,
)
```

### Multi-Task Training

Combine data from multiple tasks:

```python
# Create datasets for each task
dataset1 = DynamicsDataset(data_dir='task1_data', ...)
dataset2 = DynamicsDataset(data_dir='task2_data', ...)

# Combine examples
combined_examples = dataset1.examples + dataset2.examples
```

### Transfer Learning

Load a pretrained model and fine-tune:

```python
# Load pretrained model
network, config = load_trained_model('pretrained_model.pkl')

# Create optimizer with lower learning rate
optimizer = nnx.Optimizer(network, optax.adam(1e-4))

# Continue training...
```
