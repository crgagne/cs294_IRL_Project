import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf

import sys
sys.path.append('../crystal_quest/')
import crystal_quest_env as cq
from reward_functions import *
from dqn_inverse import *
from models import *


def main():

    # tensor flow session
    session = get_session()

    # set up reward function (specify number of features)
    reward_func = LinearRewardFunction(session)

    # set up env (make sure this is the same as ground truth)
    env = cq.Wave1Env(num_aliens=2,num_crystals=20,num_asteroids=20,
                  obs_type=3,relative_window=(25,25),
                 reward_func=reward_func)

    # random seed
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    set_global_seeds(seed)
    env.seed(seed)

    # saving
    expt_dir ='cq_irl_refactored1/'
    expt_dir ='test2'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)

    # load features for each trajectory and truncate to just the last 200 episode (which were the algorithm at optimal perf)
    crystals = np.loadtxt('../dqn/cq_gr_truth1/gym/episode_crystals2017-11-25-15:50.txt')[-200:-1]
    aliens = np.loadtxt('../dqn/cq_gr_truth1/gym/episode_alien_collisions2017-11-25-15:50.txt')[-200:-1]
    asteroids = np.loadtxt('../dqn/cq_gr_truth1/gym/episode_asteroid_collisions2017-11-25-15:50.txt')[-200:-1]
    features_demo = np.vstack((crystals,aliens,asteroids)).T

    #  randomly initialize reward
    tf.global_variables_initializer().run(session=session)
    #reward_func.set_phi(np.array([1.0,-1.0,-1.0]))
    phi_init = np.random.random(3)
    reward_func.set_phi(phi_init)

    q_func = conv_model_tiny

    num_timesteps=40000000
    num_iterations = float(num_timesteps) / 4.0

    # leanring rate
    lr_multiplier = 2.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],outside_value=5e-5 * lr_multiplier)
    optimizer = OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule)

    # Q learning is off-policy, so I actually don't need to keep this low..
    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.5),
            (num_iterations / 2, 0.001),
        ], outside_value=0.001
    )

    learn(
        env,
        q_func=q_func,
        optimizer_spec=optimizer,
        session=session,
        features_demo=features_demo,
        phi_init=phi_init,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=500000, # I think we can have a large replay buffer size.
        batch_size=32,
        gamma=0.99,
        learning_starts=50000, # and then fewer learning starts.
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=5000,    # this means every 20,000 steps
                                # this should probably be smaller too. but how many times more than the reward update freq
        reward_update_freq=10000, # this means every 40,000 steps; this also should probably be smaller
        grad_norm_clipping=10
    )
    env.close()



if __name__ == "__main__":
    main()
