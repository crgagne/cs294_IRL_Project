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
import glob

def main():

    # tensor flow session
    session = get_session()

    # set up reward function (specify number of features)
    reward_func = LinearRewardFunction(session,learning_rate=0.0001)

    # set up env (make sure this is the same as ground truth)
    env = cq.Wave1Env(num_aliens=2,num_crystals=40,num_asteroids=30,
                      obs_type=3,relative_window=(25,25),max_steps=100,
                     reward_func=reward_func,features=['crystal_captured',
                        'asteroid_collision',
                        'alien_collision'],stochastic_actions=True,choice_noise=0.15,clumping_factor=1.5,
                        num_crystal_clumps=2,
                        num_asteroid_clumps=2)

    # random seed
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    set_global_seeds(seed)
    env.seed(seed)

    #expt_dir ='cq_irl_safer_clust1.5_cn15_tiny_tmp0.1_100steps_b/'
    #expert_dir = 'cq_grt_safer_clust1.5_cn15_tiny_tmp0.1_100steps/'

    expt_dir ='cq_irl_riskier_clust1.5_cn15_tiny_tmp0.1_100steps_b/'
    expert_dir = 'cq_grt_riskier_clust1.5_cn15_tiny_tmp0.1_100steps/'

    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True,video_callable=video_schedule)


    # load expert features
    which=0
    crystals = np.loadtxt(sorted(glob.glob(expert_dir+'/gym/*crystals*'))[which])[-1000:]
    aliens = np.loadtxt(sorted(glob.glob(expert_dir+'/gym/*alien*'))[which])[-1000:]
    asteroids = np.loadtxt(sorted(glob.glob(expert_dir+'/gym/*asteroid*'))[which])[-1000:]
    features_demo = np.vstack((crystals,asteroids,aliens)).T

    #  randomly initialize reward
    tf.global_variables_initializer().run(session=session)
    #reward_func.set_phi(np.array([1.0,-1.0,-1.0]))
    phi_init = np.random.random(3)
    reward_func.set_phi(phi_init)

    q_func = conv_model_tiny

    num_timesteps=40000000
    num_iterations = float(num_timesteps) / 4.0

    # leanring rate
    lr_multiplier = 4.0

    #
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
