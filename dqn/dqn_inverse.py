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

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    # construct a second one for the  IRL samples which are sampled with no exploration
    replay_buffer2 = ReplayBuffer(replay_buffer_size, frame_history_len)

    # for logging
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    loss = -float('nan')
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

    saver = tf.train.Saver()


    for t in itertools.count():

        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        # Sample from the environment
        # XXX I'm not sure I want to pass the episode storage here..
        # Think about whether i need 2
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
                env.env.reward_func.set_phi(phi_init)
                model_initialized=True

            # update the network
            learning_rate = optimizer_spec.lr_schedule.value(t)
            q_graph.update(obs_t_batch, act_t_batch, rew_batch, obs_tp1_batch, done_mask,learning_rate)

            # update num_param updates..
            num_param_updates+=1

            # update target network parameters every N target replays (param updates)
            if num_param_updates%target_update_freq==0:
                print('updating target network')
                q_graph.update_target()

            # update reward function #
            if num_param_updates%reward_update_freq==0:
                print('updating reward function')

                #### Sample New ####
                #### Do I want to store this somewhere else?
                #### Do I need a second one?
                optimal_episode_storage = {}
                optimal_episode_storage['episode_rewards']=[]
                optimal_episode_storage['episode_asteroid_collisions']=[]
                optimal_episode_storage['episode_alien_collisions']=[]
                optimal_episode_storage['episode_crystals_captured']=[]
                optimal_episode_storage['samp_episode_prob_traj']=[]
                optimal_episode_storage['samp_episode_log_prob_traj']=[]

                for new_samp in range(10000):

                    last_obs,optimal_episode_storage = sample_env(env,
                        session,
                        q_graph,
                        last_obs,
                        replay_buffer2, # feed in second replay buffer ..
                        0.05, # greedy
                        model_initialized,
                        optimal_episode_storage)

                features_sample = np.vstack((
                    optimal_episode_storage['episode_crystals_captured'],
                    optimal_episode_storage['episode_alien_collisions'],
                    optimal_episode_storage['episode_asteroid_collisions'])).T
                #print(samp_episode_prob_traj)
                #print(samp_episode_log_prob_traj)
                # need to calculate prob_trajectories
                print('feature counts for expert')
                print(features_demo.mean(axis=0))
                print('feature counts for q-learner under this reward')
                print(features_sample.mean(axis=0))

                batch_size=25
                K = 500 # this might be too many **
                for u in range(K):
                    # get demo features
                    batch_idx = np.random.choice(range(len(features_demo)),size=batch_size)
                    batch_idx_samp = np.random.choice(range(len(features_sample)),size=batch_size)

                    # update loss function
                    loss = env.env.reward_func.update(features_demo[batch_idx,:],
                        np.vstack((features_sample[batch_idx_samp,:],features_demo[batch_idx,:])))
                        #w=np.array(samp_episode_prob_traj)[batch_idx_samp])


        ### 4. Log progress
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:

            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            savedir = get_wrapper_by_name(env, "Monitor").directory

            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

            print("Timestep %d" % (t,))
            time_steps.append(t)
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            mean_episode_rewards.append(mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            best_mean_episode_rewards.append(best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            explorations.append(exploration.value(t))
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            lrs.append(optimizer_spec.lr_schedule.value(t))
            print('mean episode rewards (50) %d' % np.mean(episode_storage['episode_rewards'][-50:]))
            print('mean episode crystals %d' % np.mean(episode_storage['episode_crystals_captured'][-50:]))
            print('mean episode alien collisions %d' % np.mean(episode_storage['episode_alien_collisions'][-50:]))
            print('mean episode asteroid collisions %d' % np.mean(episode_storage['episode_asteroid_collisions'][-50:]))
            print('learning rate %f' % env.env.reward_func.learning_rate)
            print('last irl loss %f' %loss)
            phi = session.run(env.env.reward_func.phi)
            print('phi:')
            print(phi)
            print('bias')
            bias = session.run(env.env.reward_func.b)
            print(bias)

            sys.stdout.flush()
            savename = date_string
            np.savetxt(savedir+'/mean_episode_rewards'+savename+'.txt',np.array(mean_episode_rewards))
            np.savetxt(savedir+'/best_mean_episode_rewards'+savename+'.txt',np.array(best_mean_episode_rewards))
            np.savetxt(savedir+'/time_steps'+savename+'.txt',np.array(time_steps))
            np.savetxt(savedir+'/explorations'+savename+'.txt',np.array(explorations))
            np.savetxt(savedir+'/lrs'+savename+'.txt',np.array(lrs))
            saver.save(session,savedir+'/model_weights'+savename+'.ckpt')
