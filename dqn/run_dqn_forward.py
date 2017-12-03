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
    #env = cq.Wave1Env(num_aliens=2,num_crystals=20,num_asteroids=20,
    #                  obs_type=3,relative_window=(25,25),
    #                 reward_func=reward_func,features=['crystal_captured',
    #                    'asteroid_collision',
    #                    'alien_collision',
    #                  'dist_closest_asteroid',
    #                 'dist_closest_alien'],stochastic_actions=False,choice_noise=.20)

    env = cq.Wave1Env(num_aliens=2,num_crystals=20,num_asteroids=20,
                      obs_type=3,relative_window=(25,25),
                     reward_func=reward_func,features=['crystal_captured',
                        'asteroid_collision',
                        'alien_collision'],stochastic_actions=False,choice_noise=.05)

    # random seed
    seed = 0
    set_global_seeds(seed)
    env.seed(seed)

    # saving
    expt_dir ='cq_gr_truth_more_choice_noise5_softmax/'
    #expt_dir ='test/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)

    # intialize session for the reward function
    # ends up getting re-initalized later
    tf.global_variables_initializer().run(session=session)

    # set the ground truth
    #gt_reward = np.array([2.0,-1.0,-1.0])
    gt_reward = np.array([10.0,-20.0,-20.0,0.05,0.05])
    #gt_reward = np.array([5.0,0.0,0.0])
    gt_reward = np.array([5.0,-50.0,-50.0])
    gt_reward = np.array([5.0,-1.0,-1.0])
    reward_func.set_phi(gt_reward)

    # Set up q function (imported from models)
    q_func = conv_model_small

    #
    num_timesteps=40000000

    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    # Set up a learning rate schedule
    lr_multiplier = 3.0
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
        replay_buffer_size=50000, # vs 1e6 to save memory space
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
