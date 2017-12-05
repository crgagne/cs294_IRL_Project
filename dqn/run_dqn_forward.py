import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np

import sys
sys.path.append('../crystal_quest/')
import crystal_quest_env as cq
from dqn_forward import *
from dqn_utils import *
from reward_functions import *
from models import *


def main():

    # Start tf Session
    session = get_session()

    # set up reward function
    reward_func = LinearRewardFunction(session,num_features=3)

    # set up env
    env = cq.Wave1Env(num_aliens=2,num_crystals=40,num_asteroids=30,
                      obs_type=3,relative_window=(25,25),max_steps=100,
                     reward_func=reward_func,features=['crystal_captured',
                        'asteroid_collision',
                        'alien_collision'],stochastic_actions=True,choice_noise=0.15,clumping_factor=1.5,
                        num_crystal_clumps=2,
                        num_asteroid_clumps=2)

    # random seed
    seed = 0
    set_global_seeds(seed)
    env.seed(seed)

    # saving
    #expt_dir ='cq_grt_risky_clust1.5_and_10p_cn15_soft_tiny/'
    #expt_dir ='cq_grt_safer_clust1.5_and_10p_cn15_soft_tiny/'

    expt_dir ='cq_grt_safer_clust1.5_cn15_tiny_tmp0.1_100steps/'
    expt_dir ='cq_grt_safer_wneg3_clust1.5_cn15_tiny_tmp0.1_100steps/'

    #expt_dir ='cq_grt_neut_clust1.5_cn15_tiny_tmp0.1_100steps/'
    #expt_dir ='cq_grt_riskier_clust1.5_cn15_tiny_tmp0.1_100steps/'

    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True,video_callable=video_schedule)

    # intialize session for the reward function
    # ends up getting re-initalized later
    tf.global_variables_initializer().run(session=session)

    # set the ground truth
    gt_reward = np.array([1.0,-3.0,-3.0])
    #gt_reward = np.array([1.0,-1.0,-1.0])
    #gt_reward = np.array([1.0,-0.1,-0.1])


    reward_func.set_phi(gt_reward)

    # Set up q function (imported from models)
    q_func = conv_model_tiny

    #
    num_timesteps=40000000

    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    # Set up a learning rate schedule
    lr_multiplier = 4.0
    lr_schedule = PiecewiseSchedule([
                                    (0,                   1e-4 * lr_multiplier),
                                    (num_iterations / 10, 1e-4 * lr_multiplier),
                                    (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],outside_value=5e-5 * lr_multiplier)
    # optimizer
    optimizer = OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule)


    # Set up an exploration schedule
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0), #
            (5e5, 0.1), # (1e6,0.1)
            (num_iterations / 2, 0.001),
        ], outside_value=0.001
    )

    learn(
        env,
        q_func=q_func,
        optimizer_spec=optimizer,
        session=session,
        gt_reward=gt_reward,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=500000, # vs 1e6 to save memory space
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,#500000
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=5000,
        grad_norm_clipping=10
    )

    env.close()


if __name__ == "__main__":
    main()
