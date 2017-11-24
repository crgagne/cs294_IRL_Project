
import dqn_cq
from dqn_utils import *

import sys
sys.path.append('../crystal_quest/')
import crystal_quest_env as cq

import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


def cq_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)

            # Chris' architecture
            #out = layers.convolution2d(out, num_outputs=16, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            #out = layers.convolution2d(out, num_outputs=32, kernel_size=2, stride=1, activation_fn=tf.nn.relu)
            #out = layers.convolution2d(out, num_outputs=64, kernel_size=2, stride=1, activation_fn=tf.nn.relu)

        out = layers.flatten(out)
        # print(num)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def cq_learn(env,
                session,
                num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 2.0
    lr_schedule = PiecewiseSchedule([
                                         (0,                   1e-4 * lr_multiplier),
                                         (num_iterations / 10, 1e-4 * lr_multiplier),
                                         (num_iterations / 2,  5e-5 * lr_multiplier),
                                    ],
                                    outside_value=5e-5 * lr_multiplier)
    optimizer = dqn_cq.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        #return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps
        return t >= num_timesteps


    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (1e6, 0.1),
            (num_iterations / 2, 0.001),
        ], outside_value=0.001
    )


    dqn_cq.learn(
        env,
        q_func=cq_model,
        optimizer_spec=optimizer,
        session=session,
        exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=500000,#500000
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()

def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.physical_device_desc for x in local_device_protos if x.device_type == 'GPU']

def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(seed):


    env = cq.Wave1Env(relative_window=(25,25))
    set_global_seeds(seed)

    # XXX HOW IMPORTANT IS THIS: ??
    #env.seed(seed)

    expt_dir ='cq_test/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)

    return env

def main():

    # Run training
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(seed)
    session = get_session()
    cq_learn(env, session, num_timesteps=40000000)

if __name__ == "__main__":
    main()
