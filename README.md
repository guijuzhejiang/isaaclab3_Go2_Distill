# AER-Adaptive-Gait: Unitree GO2 Velocity Locomotion with Policy Distillation

[English Version](README.md) | [中文文档](README_ZH.md)

This repository provides an open-source template and implementation for training a robust velocity-tracking locomotion policy on the Unitree Go2 quadrupedal robot over varied complex terrains. Built on **Isaac Lab v2.3.2** and **RSL-RL v5**, the repository features a full Sim-to-Real reinforcement learning pipeline using **Privileged-Information Teacher Training, Terrain Curriculum Learning, Policy Distillation (Behavior Cloning), and Student Fine-tuning**.

---

## 🎥 Training Demonstration

![](assets/demo.gif)

---

## 💡 Key Methodologies

### 1. Generalization to Hybrid Complex Terrains
To guarantee that the learnt locomotion policy generalizes well to unstructured real-world environments, the training scene is configured with `COBBLESTONE_ROAD_CFG` containing a rich mixture of terrains:
- **Sloped Terrains** (`smooth_slope`, `smooth_slope_inv`): Trains attitude control, roll/pitch adjustments.
- **Staircases/Steps** (`pyramid_stairs`, `pyramid_stairs_inv`): Trains foothold planning and swing phase leg elevation.
- **Rough Surfaces & Obstacles** (`random_rough`, `discrete_obstacles`): Induces high frequency perturbation and step clearance behavior.

### 2. Adaptive Terrain Curriculum Learning
Directly training an agent on hard terrains leads to poor sample efficiency and policy collapse. We use a **curriculum manager** (`terrain_levels_vel`):
- It evaluates the robot's walking distance within the episode relative to target velocity commands.
- Successful environments upgrade the agent to harder terrain difficulty levels (`move_up`).
- Failures or short walks downgrade the agent (`move_down`), ensuring a smooth learning curve.

### 3. Sim-to-Real Teacher-Student Distillation
The gap between simulation and the real world (Sim-to-Real gap) often prevents primary policies from deploying since they rely on unobservable variables:
- **Teacher Policy**: Observes the "Policy" group which contains *privileged* variables like local ground height scan maps (`height_scanner`) and true base linear velocities (`base_lin_vel`).
- **Student Policy**: Observes only the "Student" group containing proprioceptive data (body angular velocity, gravity projection, joint positions/velocities, joint torques, last action, velocity command inputs). No scanning or linear velocity measurement is injected.
- **Distillation Process**: The student network contains a Recurrent Neural Network (GRU) matching teacher actions under a Huber loss, which implicitly learns to estimate the latent terrain and speed information from the history of observations.

---

## 🛠️ Training & Distillation Pipeline Commands

All Python scripts are configured to run in the default conda environment:
```bash
conda activate py312_cu121
```

### Distill the Teacher into the Student Model (Go2-velocity-v1)
Run the distillation task to teach the recurrent student model using behavior cloning. You must supply the load run folder of your pre-trained teacher checkpoint:
```bash
python scripts/rsl_rl/train.py --task Go2-velocity-Distill-v0  --num_envs 4096 --load_run <teacher_run_folder> --checkpoint <teacher_model.pt>
```
*Example: `--load_run 2026-06-29_12-00-00_go2_demo --checkpoint model_10000.pt`*

### Fine-Tune the Distilled Student Policy
After distillation, you can fine-tune the distilled student network using standard PPO directly under student observations to adapt to closed-loop feedback control:
```bash
python scripts/rsl_rl/train.py --task Go2-velocity-v0 --num_envs 4096 --resume --load_run <student_distilled_run_folder> --checkpoint <student_distilled_model.pt>
```

---

## 🎮 Play & Deployment Export

To play and visualize checkpoints during training, use the evaluation script:
```bash
# Play teacher checkpoint
python scripts/rsl_rl/play.py --task Go2-velocity-Distill-v0 --load_run <teacher_run_folder> --checkpoint <model_name.pt>

# Play student checkpoint (distilled/fine-tuned)
python scripts/rsl_rl/play.py --task Go2-velocity-v0 --load_run <student_run_folder> --checkpoint <model_name.pt>
```
Playing checkpoints automatically generates optimized deployment files inside the `exported` subdirectory:
- `policy.pt` (TorchScript / JIT format)
- `policy.onnx` (ONNX representation)

---

## ⚙️ Installation

1. Install Isaac Lab by following the [official installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
2. Install this project extension in editable mode:
   ```bash
   python -m pip install -e source/isaaclab3_Go2_Distill
   ```
3. List environment tasks to verify the registration:
   ```bash
   python scripts/list_envs.py
   ```
