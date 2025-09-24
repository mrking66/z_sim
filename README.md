# Z-Sim 强化学习机器人仿真环境
1.克隆仓库:
```bash
git clone https://github.com/mrking66/z_sim.git
cd z_sim
```
    
2.创建虚拟环境并安装依赖:
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