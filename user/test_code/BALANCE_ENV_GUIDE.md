# 平衡机器人强化学习环境使用指南

## 📁 文件说明

已创建以下文件：

1. **balance_robot_env_cfg.py** - 环境配置文件
   - 定义动作/观察空间
   - 机器人和传感器配置
   - 奖励函数权重
   - 终止条件参数

2. **balance_robot_env.py** - 环境实现文件
   - 场景设置
   - 观察获取
   - 奖励计算
   - 终止判断
   - 重置逻辑

## 🔧 必须修改的部分

### 第一步：确定机器人信息

运行测试脚本查看机器人信息：

```bash
cd /home/xyz/Desktop/xluo/bipedal_rl/test
python user/test_code/test_robot_jointsNsensors.py
```

查看输出，记录：
- **Joint names**: 关节名称列表
- **Number of joints**: 关节数量
- **Body names**: 刚体名称列表（找到base_link）

### 第二步：修改配置文件

打开 `balance_robot_env_cfg.py`，按照TODO注释修改：

#### 1. 动作和观察空间 (第28-40行)

```python
# 修改为你的关节数量
action_space = 2  # 例如：2个轮子关节

# 计算观察空间维度：
# IMU(10) + 关节位置(n) + 关节速度(n)
# 如果 n=2: observation_space = 10 + 2 + 2 = 14
observation_space = 14
```

#### 2. 机器人关节名称 (第133行)

```python
# 替换为实际的关节名称
controlled_joint_names = ["left_wheel_joint", "right_wheel_joint"]
```

#### 3. IMU传感器位置 (第118行)

```python
# 替换为你的base_link名称
prim_path="/World/envs/env_.*/Robot/base_link"
```

#### 4. 执行器参数 (第97-107行)

```python
# 根据电机规格修改扭矩限制
effort_limit=80.0  # 单位：牛顿米
```

### 第三步：修改环境实现

打开 `balance_robot_env.py`，按照TODO注释修改：

#### 1. 获取控制关节索引 (第28-36行)

方案A - 按名称获取：
```python
self._left_wheel_idx, _ = self.robot.find_joints("left_wheel_joint")
self._right_wheel_idx, _ = self.robot.find_joints("right_wheel_joint")
self._controlled_joint_indices = [self._left_wheel_idx[0], self._right_wheel_idx[0]]
```

方案B - 控制所有关节：
```python
self._controlled_joint_indices = list(range(self.robot.num_joints))
```

#### 2. 确认观察向量 (第81-115行)

确保观察向量的维度与配置文件中的 `observation_space` 匹配。

默认包含：
- IMU线性加速度: 3维
- IMU角速度: 3维
- IMU姿态四元数: 4维
- 关节位置: n维
- 关节速度: n维

总计: 10 + n + n 维

#### 3. 自定义奖励函数 (第117-180行)

根据你的任务目标调整奖励项和权重。默认包含：
- 存活奖励
- 姿态奖励（保持直立）
- 速度惩罚
- 关节速度惩罚
- 动作平滑
- 扭矩惩罚
- 终止惩罚

## 📝 完整修改步骤示例

假设你的机器人有2个轮子关节，名称为 "joint_left_wheel" 和 "joint_right_wheel"：

### 修改 balance_robot_env_cfg.py:

```python
# 第29行
action_space = 2

# 第40行
observation_space = 14  # 10(IMU) + 2(pos) + 2(vel)

# 第118行
prim_path="/World/envs/env_.*/Robot/base_link"  # 确认base_link名称

# 第133行
controlled_joint_names = ["joint_left_wheel", "joint_right_wheel"]
```

### 修改 balance_robot_env.py:

```python
# 第28-32行
def __init__(self, cfg: BalanceRobotEnvCfg, render_mode: str | None = None, **kwargs):
    super().__init__(cfg, render_mode, **kwargs)
    
    # 获取轮子关节索引
    self._left_wheel_idx, _ = self.robot.find_joints("joint_left_wheel")
    self._right_wheel_idx, _ = self.robot.find_joints("joint_right_wheel")
    self._controlled_joint_indices = [self._left_wheel_idx[0], self._right_wheel_idx[0]]
```

## 🚀 测试环境

### 1. 快速语法检查

```bash
cd /home/xyz/Desktop/xluo/bipedal_rl/test
python -c "from source.test.test.tasks.direct.test.balance_robot_env_cfg import BalanceRobotEnvCfg; print('Config OK')"
python -c "from source.test.test.tasks.direct.test.balance_robot_env import BalanceRobotEnv; print('Env OK')"
```

### 2. 创建测试脚本

创建 `user/test_code/test_balance_env.py`:

```python
"""测试平衡机器人环境"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from source.test.test.tasks.direct.test.balance_robot_env_cfg import BalanceRobotEnvCfg_PLAY
from source.test.test.tasks.direct.test.balance_robot_env import BalanceRobotEnv

def main():
    env_cfg = BalanceRobotEnvCfg_PLAY()
    env = BalanceRobotEnv(cfg=env_cfg)
    
    print(f"环境创建成功！")
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"环境数量: {env.num_envs}")
    
    # 运行几个step测试
    env.reset()
    for i in range(100):
        actions = env.action_space.sample()
        obs, rewards, dones, truncated, info = env.step(actions)
        if i % 10 == 0:
            print(f"Step {i}: Reward mean = {rewards.mean().item():.3f}")
    
    print("测试完成！")

if __name__ == "__main__":
    main()
    simulation_app.close()
```

运行测试：
```bash
python user/test_code/test_balance_env.py --num_envs 1
```

## 🎯 训练环境

修改好环境后，需要在 `__init__.py` 中注册：

编辑 `source/test/test/tasks/__init__.py`:

```python
import gymnasium as gym
from . import direct

# 注册环境
gym.register(
    id="Template-Balance-Robot-Direct-v0",
    entry_point="test.tasks.direct.test:BalanceRobotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "test.tasks.direct.test:BalanceRobotEnvCfg",
    },
)
```

然后就可以用RL库训练了：

```bash
# 使用 RSL-RL
python scripts/rsl_rl/train.py --task Template-Balance-Robot-Direct-v0

# 使用 Stable-Baselines3
python scripts/sb3/train.py --task Template-Balance-Robot-Direct-v0
```

## 🐛 常见问题

### Q1: 维度不匹配错误
```
AssertionError: 观察维度不匹配！期望 14，实际 16
```
**解决**: 检查 `_get_observations()` 中拼接的张量维度，确保与 `observation_space` 一致

### Q2: 找不到关节
```
RuntimeError: Joint 'xxx' not found
```
**解决**: 运行 test_robot_jointsNsensors.py 查看正确的关节名称

### Q3: IMU传感器不工作
```
AttributeError: 'NoneType' object has no attribute 'data'
```
**解决**: 检查 IMU 的 prim_path 是否正确，base_link名称是否匹配

### Q4: 机器人穿地
**解决**: 调整 `init_state.pos` 的z值，抬高初始位置

## 📚 参考资料

- Isaac Lab 文档: https://isaac-sim.github.io/IsaacLab/
- Direct RL 环境示例: `IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/`
- 你的测试代码: `user/test_code/test_robot_jointsNsensors.py`

## 💡 下一步

1. ✅ 修改配置文件中的TODO项
2. ✅ 修改环境实现中的TODO项
3. ✅ 运行测试脚本验证
4. ✅ 注册环境到 `__init__.py`
5. ✅ 开始训练！

根据训练效果调整：
- 奖励函数权重
- 动作缩放因子
- 终止条件阈值
- 观察空间（添加或删除观测）
