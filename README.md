# Z-Sim 强化学习机器人仿真环境1.  Clone this repository:

    git clone https://github.com/your-username/your-repo-name.git

这是一个用于机器人强化学习研究的仿真环境项目。    cd your-repo-name

2.  Set up a virtual environment and install dependencies.

## 安装说明    python -m venv rl_env

    # On Windows

1. 克隆此仓库：        .\rl_env\Scripts\activate

```bash    # On Linux/MacOS

git clone https://github.com/mrking66/z_sim.git        source rl_env/bin/activate

cd z_sim    pip install -r requirements.txt

```3.  Run the code!

    python zeroth_train.py
2. 设置虚拟环境并安装依赖：
```bash
# 创建虚拟环境
python -m venv rl_env

# Windows系统激活虚拟环境
.\rl_env\Scripts\activate

# Linux/MacOS系统激活虚拟环境
source rl_env/bin/activate

# 安装依赖包
pip install -r requirements.txt
```

3. 运行代码：
```bash
python zeroth_train.py
```

## 项目结构

```
z_sim/
├── mechanism/           # 机器人模型文件
│   ├── meshes/         # 机器人各部件的3D模型文件
│   ├── humanoid.mjcf   # MuJoCo模型文件
│   ├── joints.py       # 关节定义
│   └── robot_fixed.xml # 机器人配置文件
├── zeroth_env.py       # 仿真环境定义
├── zeroth_train.py     # 训练脚本
└── requirements.txt    # 项目依赖
```