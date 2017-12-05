
"""Deep Q learning graph

"""


from dqn_utils import *

class QGraph():

    def __init__(self,
        input_shape,
        num_actions,
        q_func,
        session,gamma,optimizer_spec,grad_norm_clipping,temp=0.1):
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.session=session
        self.q_func=q_func
        self.gamma=gamma
        self.optimizer_spec = optimizer_spec
        self.grad_norm_clipping=grad_norm_clipping
        self.obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
        self.act_t_ph              = tf.placeholder(tf.int32,   [None])
        self.rew_t_ph              = tf.placeholder(tf.float32, [None])
        self.obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
        self.done_mask_ph          = tf.placeholder(tf.float32, [None])
        #self.temp = tf.placeholder(shape=None,dtype=tf.float32)
        self.temp = temp
        # casting to float on GPU ensures lower data transfer times.
        self.obs_t_float   = tf.cast(self.obs_t_ph,   tf.float32) / 255.0
        self.obs_tp1_float = tf.cast(self.obs_tp1_ph, tf.float32) / 255.0


    def build_train(self,soft=False):

        # q values for current s,a
        self.q_val_t = self.q_func(self.obs_t_float, self.num_actions, scope="q_func", reuse=False)

        #self.q_probs = tf.exp(self.q_val_t ) / tf.reduce_sum(tf.exp(self.q_val_t),axis=1) #ah.. it was reducing it across batch and val
        self.q_probs = tf.nn.softmax(self.q_val_t/self.temp,dim=1) # #ah.. it was reducing it across batch and val

        print(np.shape(self.q_probs))

        # q value for action taken
        self.q_val_t_selected = tf.reduce_sum(self.q_val_t*tf.one_hot(self.act_t_ph,self.num_actions),axis=1)

        # q values for next time step
        self.q_val_tp1 = self.q_func(self.obs_tp1_float, self.num_actions, scope="target_q_func", reuse=False)

        # Max next q value (masked to 0 if done)
        self.q_val_tp1_best_masked = tf.reduce_max(self.q_val_tp1,reduction_indices=[1])*(1.0-self.done_mask_ph) ### done mask ****

        # Soft-max next Q-value (removing . )
        # can't believe they have this function built in (this will hopefully solve my underflow/overflow issues)
        #self.q_val_tp1_softmax_masked = tf.reduce_logsumexp(self.q_val_tp1,axis=1)*(1.0-self.done_mask_ph)

        #if soft:
        #    self.q_val_t_selected_target = self.rew_t_ph + self.gamma*self.q_val_tp1_softmax_masked
        #else:
        self.q_val_t_selected_target = self.rew_t_ph + self.gamma*self.q_val_tp1_best_masked


        # temporal difference error
        self.td_error = self.q_val_t_selected-tf.stop_gradient(self.q_val_t_selected_target)
        self.loss_huber = huber_loss2(self.td_error)
        self.total_error = tf.reduce_mean(self.loss_huber) # not sure this is necessary

        # collect all variables associated with learning network
        self.q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')

        # collect all variables associated with target network
        self.target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

        # construct optimization op (with gradient clipping)
        self.learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
        optimizer = self.optimizer_spec.constructor(learning_rate=self.learning_rate, **self.optimizer_spec.kwargs)
        self.train_fn = minimize_and_clip(optimizer, self.total_error,
                 var_list=self.q_func_vars, clip_val=self.grad_norm_clipping)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_fn = []
        for var, var_target in zip(sorted(self.q_func_vars,        key=lambda v: v.name),
                               sorted(self.target_q_func_vars, key=lambda v: v.name)):
                               update_target_fn.append(var_target.assign(var))
        self.update_target_fn = tf.group(*update_target_fn)

    def update(self,obs_t_batch,act_t_batch,rew_batch,obs_tp1_batch,done_mask,learning_rate):

        # get total error (needed to set the learning rate)
        feed_dict = {self.obs_t_ph:obs_t_batch,
            self.act_t_ph:act_t_batch,
            self.rew_t_ph:rew_batch,
            self.obs_tp1_ph:obs_tp1_batch,
            self.done_mask_ph:done_mask}

        realized_total_error=self.session.run(self.total_error,feed_dict=feed_dict)

        feed_dict = {self.obs_t_ph:obs_t_batch,
                self.act_t_ph:act_t_batch,
                self.rew_t_ph:rew_batch,
                self.obs_tp1_ph:obs_tp1_batch,
                self.done_mask_ph:done_mask,
                self.total_error:realized_total_error,
                self.learning_rate:learning_rate}
        # traing the network
        self.session.run(self.train_fn,feed_dict=feed_dict)

    def update_target(self):
        self.session.run(self.update_target_fn)



    def act(self,recent_history,soft=True):
        '''returns action by softmax'''
        q_probs = self.session.run(self.q_probs,{self.obs_t_ph:np.expand_dims(recent_history,axis=0)})[0]
        qp = np.random.choice(q_probs,p=q_probs)
        action = np.argmax(q_probs == qp)

        return(action,q_probs)
