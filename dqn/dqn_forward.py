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
from models import *
import time
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def learn(env,
          q_func,
          optimizer_spec,
          session,
          gt_reward,
          exploration=LinearSchedule(1000000, 0.1),
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10):

    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

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

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)

    num_actions = env.action_space.n

    # build Q learner Graph
    #soft=False # push above
    soft=True
    q_graph = QGraph(input_shape,num_actions,q_func,session,gamma,optimizer_spec,grad_norm_clipping)
    q_graph.build_train(soft=soft)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    #################
    # Logging Setup #
    ################
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 1000 #10000
    date_string = time.strftime("%Y-%m-%d-%H:%M")
    best_mean_episode_rewards=[]
    mean_episode_rewards=[]
    time_steps = []
    lrs = []
    explorations = []

    #
    episode_storage = {}
    episode_storage['episode_rewards']=[]
    episode_storage['episode_asteroid_collisions']=[]
    episode_storage['episode_alien_collisions']=[]
    episode_storage['episode_crystals_captured']=[]
    episode_storage['episode_prob_traj']=[]
    episode_storage['episode_exp_rew']=[]
    episode_storage['prob_act']=[]
    saver = tf.train.Saver()

    ########################
    # Q-learning Algorithm #
    ########################
    for t in itertools.count():

        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        # Sample from the environment
        epsilon = exploration.value(t)
        last_obs,episode_storage = sample_env(env,
            session,
            q_graph,
            last_obs,
            replay_buffer,
            epsilon,
            model_initialized,
            episode_storage)

        # Update Q Values every 4 actual samples #
        if (t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size)):

            # sample batch from replay buffer
            obs_t_batch, act_t_batch, rew_batch, obs_tp1_batch, done_mask = replay_buffer.sample(batch_size)

            # initialize network if this is the first update
            if model_initialized==False:
                initialize_interdependent_variables(session, tf.global_variables(), {
                        q_graph.obs_t_ph: obs_t_batch,
                        q_graph.obs_tp1_ph: obs_tp1_batch})
                q_graph.update_target()

                # re-set ground truth reward
                env.env.reward_func.set_phi(gt_reward)
                model_initialized=True

            # update the network
            learning_rate = optimizer_spec.lr_schedule.value(t)
            q_graph.update(obs_t_batch, act_t_batch, rew_batch, obs_tp1_batch, done_mask,learning_rate)

            # update num_param updates.
            num_param_updates+=1

            # update target network parameters every N target replays (param updates)
            if num_param_updates%target_update_freq==0:
                print('updating target network')
                q_graph.update_target()

        # Log progress
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:

            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            savedir = get_wrapper_by_name(env, "Monitor").directory

            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

            print('')
            print("Timestep %d" % (t,))
            time_steps.append(t)
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            mean_episode_rewards.append(mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            best_mean_episode_rewards.append(best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            #print("exploration %f" % exploration.value(t))
            print("exploration soft")
            explorations.append(exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            lrs.append(optimizer_spec.lr_schedule.value(t))
            print('mean episode rewards (50) %d' % np.mean(episode_storage['episode_rewards'][-50:]))
            print('mean episode crystals %d' % np.mean(episode_storage['episode_crystals_captured'][-50:]))
            print('mean episode alien collisions %d' % np.mean(episode_storage['episode_alien_collisions'][-50:]))
            print('mean episode asteroid collisions %d' % np.mean(episode_storage['episode_asteroid_collisions'][-50:]))

            sys.stdout.flush()
            savename = date_string
            np.savetxt(savedir+'/mean_episode_rewards'+savename+'.txt',np.array(mean_episode_rewards))
            np.savetxt(savedir+'/best_mean_episode_rewards'+savename+'.txt',np.array(best_mean_episode_rewards))
            np.savetxt(savedir+'/time_steps'+savename+'.txt',np.array(time_steps))
            np.savetxt(savedir+'/explorations'+savename+'.txt',np.array(explorations))
            np.savetxt(savedir+'/lrs'+savename+'.txt',np.array(lrs))
            for k in episode_storage:
                np.savetxt(savedir+'/'+k+savename+'.txt',np.array(episode_storage[k]))
            saver.save(session,savedir+'/model_weights'+savename+'.ckpt')
