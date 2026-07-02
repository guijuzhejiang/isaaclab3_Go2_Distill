# AER-Adaptive-Gait: Unitree GO2 复杂地形速度跟随与策略蒸馏项目

[English Version](README.md) | [中文文档](README_ZH.md)

本项目基于 Isaac Lab 2.3.2 模拟框架与 RSL-RL v5 强化学习库，实现了 Unitree Go2 四足机器人在复杂未知地形环境下的鲁棒速度跟随控制。为了解决模拟到真实世界（Sim-to-Real）的跨越问题，项目采用了**特权信息教师网络训练、自适应课程学习、策略蒸馏（Distillation）以及学生网络独立微调（Fine-tuning）**的完整技术管线。

---

## 🎥 训练与演示视频

![](assets/demo.gif)

---

## 💡 核心设计与技术亮点

### 1. 混合复杂地形训练 (Generalization to Complex Terrains)
为了使策略具有极强的泛化能力，我们设计了包含多种复杂单元的地形生成器 `COBBLESTONE_ROAD_CFG`。地形混合了以下场景：
- **斜坡地形** (`smooth_slope` 与 `smooth_slope_inv`)：进行俯仰与横滚角度的姿态控制过渡。
- **台阶/楼梯** (`pyramid_stairs` 与 `pyramid_stairs_inv`)：训练机器人的越障与抬腿高度适应性。
- **粗糙高度场** (`random_rough` 与 `discrete_obstacles`)：模拟真实道路的不规则扰动和离散障碍块。

通过大规模的高动态高度场混合训练，极大提升了模型推广到真实世界复杂地表时的鲁棒性。

### 2. 自适应课程学习 (Curriculum Learning)
直接在最难的地形上学习会导致强化学习策略难以收敛。为此，我们使用了**地形难度课程学习**：
- 系统依据机器人在当前随机速度指令下走过的水平距离判断运动表现。
- 当机器人能够在当前难度关卡走得足够远并保持稳定时，课程管理器（`terrain_levels_vel`）会将其提升（`move_up`）到更难的地形关卡。
- 若发生跌倒或在限制时间内位移过小，则会退回（`move_down`）到简单关卡，以此平滑学习曲线，大幅提升训练收敛效率。

### 3. 特权信息蒸馏 (Teacher-Student Distillation)
为解决 Sim-to-Real 过程中传感器噪声和部分物理量（如绝对线速度、精细地形高度）在真实世界无法直接获取的问题：
- **教师模型（Teacher Policy）**：具备“特权信息”（Privileged Information），包括来自高度扫描传感器的局部地图高度 `height_scanner` 和完美的本体绝对线速度 `base_lin_vel`。
- **学生模型（Student Policy）**：去除了所有特权信息，输入仅包含机身角速度、重力投影投影、关节位置/速度偏差、关节出力力矩、上一时刻动作指令以及速度给定。
- **蒸馏过程**：学生网络采用循环神经网络结构（GRU）来隐式估计时序中的隐含状态（如地形高度偏差和线速度），并以 Hubers 损失函数拟合教师网络的动作分布输出。

---

## 🛠️ 训练与蒸馏工作流命令

### 第一步：训练拥有特权信息的教师模型 (Teacher Training)
首先训练一个能够完美适应复杂地形的教师智能体：
```bash
python scripts/rsl_rl/train.py --task Go2-velocity-v0 --num_envs 4096
```
*提示：训练好的模型将自动保存在 `logs/rsl_rl/go2_demo` 目录下。*

### 第二步：将教师网络蒸馏至学生网络 (Policy Distillation)
使用无特权信息的学生网络对教师网络进行行为克隆学徒训练。运行时需通过 `--load_run` 和 `--checkpoint` 指定已训练完毕的教师模型：
```bash
# 执行蒸馏训练任务（对应 Go2-velocity-v1 任务，该任务采用 DistillationRunner）
python scripts/rsl_rl/train.py --task Go2-velocity-v1 --num_envs 4096 --load_run <teacher_run_folder> --checkpoint <teacher_model_name.pt>
```
*例如： `--load_run 2026-06-29_12-00-00_go2_demo --checkpoint model_10000.pt`*

### 第三步：对蒸馏后的学生网络进行独立微调 (Student Fine-tuning)
蒸馏结束后，为进一步优化学生网络在无特权观测下的端到端表现，可以在学生观测空间下对其进行闭环 PPO 独立微调。
```bash
# 以微调模式继续在无特权传感器下训练学生模型
python scripts/rsl_rl/train.py --task Go2-velocity-v0 --num_envs 4096 --resume --load_run <student_distilled_run_folder> --checkpoint <student_distilled_model_name.pt>
```

---

## 🎮 推理与测试 (Play / Inference)

您可以使用 `play.py` 脚本以可视化的方式评估训练或蒸馏好的策略：
```bash
# 评估特权教师模型
python scripts/rsl_rl/play.py --task Go2-velocity-v0 --load_run <teacher_run_folder> --checkpoint <model_name.pt>

# 评估蒸馏或微调后的学生模型
python scripts/rsl_rl/play.py --task Go2-velocity-v1 --load_run <distilled_student_run_folder> --checkpoint <model_name.pt>
```
运行该命令会同时自动将策略导出为能够在实体控制部署中读取的 JIT 格式 (`policy.pt`) 与 ONNX 格式 (`policy.onnx`)，输出在 `exported` 文件夹中。

---

## ⚙️ 项目依赖与安装

1. 按照 [Isaac Lab 官方指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 正确配置开发环境。
2. 以外部可编辑模式安装 go2 演示包：
   ```bash
   python -m pip install -e source/go2_demo
   ```
3. 验证任务是否已成功注册：
   ```bash
   python scripts/list_envs.py
   ```
