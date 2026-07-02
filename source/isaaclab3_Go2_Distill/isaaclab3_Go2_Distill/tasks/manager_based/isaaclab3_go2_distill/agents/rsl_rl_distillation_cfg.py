# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)


#训练命令：python scripts/rsl_rl/train.py --task Go2-velocity-Distill-v0 --num_envs 400 --load_run dir --checkpoint model_xx.pt
@configclass
class DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 1000
    experiment_name = "go2_distillation"
    #在go2_demo_velocity.py中定义的policy是教师网络"teacher"，定义的student是现在要学习的学生网络"policy"
    obs_groups = {
        "teacher": ["policy"],
        "student": ["student"],
    }
    student = RslRlRNNModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="gru",
        rnn_hidden_dim=247,
        rnn_num_layers=1,
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=24,
        max_grad_norm=1.0,
        loss_type="huber",
    )