# Z-Sim: Reinforcement Learning Robot Simulation Environment

A simulation environment for robotics reinforcement learning research.

## Installation Guide

1. Clone the repository:
```bash
git clone https://github.com/mrking66/z_sim.git
cd z_sim
```

2. Set up virtual environment and install dependencies:
```bash
# Create virtual environment
python -m venv rl_env

# Activate virtual environment on Windows
.\rl_env\Scripts\activate

# Activate virtual environment on Linux/MacOS
source rl_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. Run the code:
```bash
python mujoco_train.py
```

## Project Structure

```
z_sim/
├── mechanism/           # Robot model files
│   ├── meshes/         # 3D model files for robot parts
│   ├── humanoid.mjcf   # MuJoCo model file
│   ├── joints.py       # Joint definitions
│   └── robot_fixed.xml # Robot configuration file
├── mujoco_env.py       # Simulation environment definition
├── mujoco_train.py     # Training script
└── requirements.txt    # Project dependencies
```