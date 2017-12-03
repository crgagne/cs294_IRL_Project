import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
from dqn_graph import *
import time
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def learn(env,
          q_func,
          optimizer_spec,
          session,
          features_demo,
          phi_init,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          reward_update_freq=1000,
          grad_norm_clipping=10):

    """Run IRL Algorithm.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    # Input Dimensions
    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)

    num_actions = env.action_space.n

    # build Q learner Graph
    soft=False # push above
    q_graph = QGraph(input_shape,num_actions,q_func,session,gamma,optimizer_spec,grad_norm_clipping)
    q_graph.build_train(soft=soft)


    date_string = time.strftime("%Y-%m-%d-%H:%M")

    # Initialize Graph
    initialize_interdependent_variables(session, tf.global_variables(), {
            q_graph.obs_t_ph: np.random.random((input_shape))[np.newaxis,:],
            q_graph.obs_tp1_ph: np.random.random((input_shape))[np.newaxis,:]})
    q_graph.update_target()
    env.env.reward_func.set_phi(phi_init)

    # Start Algorithm
    last_obs = env.reset()
    outer_loop=60
    for i in range(outer_loop):

        # Reset replay buffer #
        # (so that you only train DQN with s,a,rewards from latest reward function #
        replay_buffer = ReplayBuffer(50000, frame_history_len)

        # Reset feature counts ??
        episode_storage = {}
        episode_storage['episode_rewards']=[]
        episode_storage['episode_asteroid_collisions']=[]
        episode_storage['episode_alien_collisions']=[]
        episode_storage['episode_crystals_captured']=[]
        episode_storage['episode_prob_traj']=[]
        episode_storage['episode_exp_rew']=[]
        episode_storage['prob_act']=[]

        # sample from policy (or randomly if not initialized)
        for s in range(50000):
            epsilon = 0.2
            last_obs,episode_storage = sample_env(env,
                session,
                q_graph,
                last_obs,
                replay_buffer,
                epsilon,
                True,
                episode_storage)

        # calculate importance weights
        # how likely was each trajectory according to current policy
        # what was the exponential of the reward w.r.t the current reward function.
        # these should eventually converge to 1 because in soft-optimality
        # the prob of a trajectory = exponential of reward.
        #print(episode_storage)


        # Update reward samples (features)
        batch_size=25
        print('updating reward function')
        for ur in range(200):

            # get mini batch of expert samples
            batch_idx = np.random.choice(range(len(features_demo)),size=batch_size)
            batch_features_demo = features_demo[batch_idx,:]

            # get mini batch of newly sampled soft-optimal policy samples.
            batch_idx_samp = np.random.choice(range(len(episode_storage)),size=batch_size)

            batch_features_samp = np.vstack((
                np.array(episode_storage['episode_crystals_captured'])[batch_idx_samp],
                np.array(episode_storage['episode_asteroid_collisions'])[batch_idx_samp],
                np.array(episode_storage['episode_alien_collisions'])[batch_idx_samp])).T

            # get mini batch of importance weights
            #w = np.array(episode_storage['episode_exp_rew'])[batch_idx_samp]/np.array(episode_storage['episode_prob_traj'][batch_idx_samp])
            #w = np.array(episode_storage['episode_rewards'])[batch_idx_samp]-np.array(episode_storage['episode_prob_traj'])[batch_idx_samp] # its log prob actually
            w = np.ones(batch_size)
            loss = env.env.reward_func.update(batch_features_demo,
                batch_features_samp,w)
        print('batch feature counts expert')
        print(batch_features_demo.mean(axis=0))
        print('batch features counts samples')
        print(batch_features_samp.mean(axis=0))

        # Update Q function based on same set of samples
        # This is Chelsea's order - does it matter?
        print('updating q function')
        for uq in range(4000):

            # sample batch from replay buffer
            obs_t_batch, act_t_batch, rew_batch, obs_tp1_batch, done_mask = replay_buffer.sample(batch_size)

            # update the network
            #learning_rate = optimizer_spec.lr_schedule.value(t)
            learning_rate = 0.0002
            q_graph.update(obs_t_batch, act_t_batch, rew_batch, obs_tp1_batch, done_mask,learning_rate)

        # Update Target Network
        print('updating target network')
        q_graph.update_target()

        # Logging
        savedir = get_wrapper_by_name(env, "Monitor").directory
        print("Outer Iteration %d" % (i,))
        print('mean episode rewards (50) %d' % np.mean(episode_storage['episode_rewards'][-50:]))
        print('mean episode crystals %d' % np.mean(episode_storage['episode_crystals_captured'][-50:]))
        print('mean episode alien collisions %d' % np.mean(episode_storage['episode_alien_collisions'][-50:]))
        print('mean episode asteroid collisions %d' % np.mean(episode_storage['episode_asteroid_collisions'][-50:]))
        print('last irl loss %f' %loss)
        print('phi:')
        print(session.run(env.env.reward_func.phi))
        print('bias')
        print(session.run(env.env.reward_func.b))

        sys.stdout.flush()
        savename = date_string
