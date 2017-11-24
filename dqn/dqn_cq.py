import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
import time
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def learn(env,
          q_func,
          optimizer_spec,
          session,
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

    print('learning_starts')
    print(learning_starts)

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        print('env observation space')
        print(env.observation_space.shape)
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)

    num_actions = env.action_space.n

    # set up placeholders
    # XXX Make sure these are the right types for act, rew, done,
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))

    done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0


    # q values for current s,a
    q_val_t = q_func(obs_t_float, num_actions, scope="q_func", reuse=False)

    # q value for action taken
    q_val_t_selected = tf.reduce_sum(q_val_t*tf.one_hot(act_t_ph,num_actions),axis=1)

    # q values for next time step
    q_val_tp1 = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False)

    # best next q value (masked to 0 if done)
    q_val_tp1_best_masked = tf.reduce_max(q_val_tp1,reduction_indices=[1])*(1.0-done_mask_ph) ### done mask ****

    # target (combining reward + next best q-value)
    q_val_t_selected_target = rew_t_ph + gamma*q_val_tp1_best_masked

    # temporal difference error
    # put a stop gradient in targets, but I'm not sure if that's necessary because they have different scopes (??)
    td_error = q_val_t_selected-tf.stop_gradient(q_val_t_selected_target)

    loss_huber = huber_loss2(td_error) # updated with tf.where()

    # average loss over batch (??)
    total_error = tf.reduce_mean(loss_huber)

    # collect all variables associated with learning network
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')

    # collect all variables associated with target network
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 1000 #10000
    print('last obs shape')
    print(last_obs.shape)
    print('')

    # for logging
    date_string = time.strftime("%Y-%m-%d-%H:%M")
    best_mean_episode_rewards=[]
    mean_episode_rewards=[]
    mean_episode_crystals=[]
    mean_episode_deaths=[]
    episode_deaths=[]
    episode_crystals=[]
    explorations=[]
    time_steps = []
    lrs = []
    saver = tf.train.Saver()

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        # store previous frame or current frame into replay buffer?
        idx = replay_buffer.store_frame(last_obs)

        # Gets 4 last obs to feed to network..
        recent_history= replay_buffer.encode_recent_observation()

        # epislon greedy exploration (DOUBLE CHECK ON EP)
        epislon = exploration.value(t)
        if not model_initialized or random.random()<epislon:
            action = env.action_space.sample()
            # action = 0 if np.random.random() < .5 else np.random.randint(low=1,high=num_actions)

        else:
            realized_q_val_t = session.run(q_val_t,{obs_t_ph:np.expand_dims(recent_history,axis=0)})
            action = np.argmax(realized_q_val_t)

        # take a step
        obs, reward, done, info = env.step(action)

        # store effect of action on last obs
        replay_buffer.store_effect(idx,action,reward,done)

        if done:
            episode_crystals.append(env.env.episode_crystals)
            episode_deaths.append(env.env.episode_deaths)
            obs = env.reset()


        last_obs = obs.copy()

        ##########
        # Update #
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            if t%1000==0:
                print(t)

            # sample (s,a,s',r) experiences from the replay buffer
            obs_t_batch, act_t_batch, rew_batch, obs_tp1_batch, done_mask = replay_buffer.sample(batch_size)

            # initialize network with the appropriate set of variables/parameters
            if model_initialized==False:
                initialize_interdependent_variables(session, tf.global_variables(), {
                        obs_t_ph: obs_t_batch,
                        obs_tp1_ph: obs_tp1_batch,
                    })
                session.run(update_target_fn)
                model_initialized=True #do I have to set this?

            # get total error (needed to set the learning rate)
            feed_dict = {obs_t_ph:obs_t_batch,
            act_t_ph:act_t_batch,
            rew_t_ph:rew_batch,
            obs_tp1_ph:obs_tp1_batch,
            done_mask_ph:done_mask}

            realized_total_error=session.run(total_error,feed_dict=feed_dict)

            feed_dict = {obs_t_ph:obs_t_batch,
                        act_t_ph:act_t_batch,
                        rew_t_ph:rew_batch,
                        obs_tp1_ph:obs_tp1_batch,
                        done_mask_ph:done_mask,
                        total_error:realized_total_error,
                        learning_rate:optimizer_spec.lr_schedule.value(t)}

            # traing the network
            session.run(train_fn,feed_dict=feed_dict)

            # update num_param updates..
            num_param_updates+=1

            # update target network parameters every N target replays (param updates)
            if num_param_updates%target_update_freq==0:
                session.run(update_target_fn)
                print('updating target network')

        ### 4. Log progress

        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:

            # print out some other stuff #
            # print('obs_t_batch')
            # print(obs_t_batch)
            # print('obs_tp1_batch')
            # print(obs_tp1_batch)
            # print('act_t_batch')
            # print(act_t_batch)
            # print('rew_batch')
            # print(rew_batch)
            # print('obs_t_batch')
            # print(obs_t_batch)
            # print('done mask')
            # print(done_mask)
            #
            # print('realized_q_val_t')
            # print(realized_q_val_t)
            realized_total_error=session.run(total_error,feed_dict=feed_dict)
            # print('realized_total_error')
            # print(realized_total_error)
            #
            # print('obs_t_ph')
            # print(session.run(obs_t_ph,feed_dict=feed_dict))
            # print(session.run(obs_t_float,feed_dict=feed_dict))
            # print('obs_tp1_ph')
            # print(session.run(obs_tp1_ph,feed_dict=feed_dict))
            # print(session.run(obs_tp1_float,feed_dict=feed_dict))
            #
            # print('act_t_ph')
            # print(session.run(act_t_ph,feed_dict=feed_dict))
            #
            # print('rew_t_ph')
            # print(session.run(rew_t_ph,feed_dict=feed_dict))
            #
            # print('q_val_t ')
            # print(session.run(q_val_t ,feed_dict=feed_dict))
            #
            # print('q_val_t_selected ')
            # print(session.run(q_val_t_selected ,feed_dict=feed_dict))
            #
            # print('q_val_tp1_best_masked')
            # print(session.run(q_val_tp1_best_masked,feed_dict=feed_dict))
            #
            # print('td_error')
            # print(session.run(td_error,feed_dict=feed_dict))


            episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
            savedir = get_wrapper_by_name(env, "Monitor").directory


            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
                mean_episode_crystal = np.mean(episode_crystals[-100:])
                mean_episode_death = np.mean(episode_deaths[-100:])
                # this is not right ...because i'm not counting per ep but episode death are long
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
            print('mean episode crystals %d' % mean_episode_crystal)
            print('mean episode deaths %d' % mean_episode_death)


            sys.stdout.flush()
            savename = date_string
            #print(np.array(mean_episode_rewards))
            np.savetxt(savedir+'/mean_episode_rewards'+savename+'.txt',np.array(mean_episode_rewards))
            np.savetxt(savedir+'/best_mean_episode_rewards'+savename+'.txt',np.array(best_mean_episode_rewards))
            np.savetxt(savedir+'/time_steps'+savename+'.txt',np.array(time_steps))
            np.savetxt(savedir+'/explorations'+savename+'.txt',np.array(explorations))
            np.savetxt(savedir+'/lrs'+savename+'.txt',np.array(lrs))
            saver.save(session,savedir+'/model_weights'+savename+'.ckpt')
