
import tensorflow as tf
import numpy as np


class LinearRewardFunction():
    '''Reward function gets passed to Crystal Quest env.
     It calculates reward on env features given current parameters.
     Parameters can also be updated with gradient descent.
     The reward function is:
        r = phi_1*crystal_acquired + phi_2*asteroid_hit + phi3*alien_hit
     Possible extensions:
        distance to alien and asteroid.
    '''
    def __init__(self,
           sess,
           learning_rate=0.0001,
           num_features=3,
           ):

        self.sess = sess
        self.num_features = num_features
        self.learning_rate=learning_rate;
        self.features = tf.placeholder(tf.float32, [None] + [self.num_features]) # first dim is for batch
        self.phi = tf.Variable(tf.zeros([self.num_features]))
        self.b = tf.Variable(tf.zeros([1]))
        self.reward = tf.matmul(self.features,tf.expand_dims(self.phi,1))


        self.batch_demo_features=tf.placeholder(tf.float32, [None] + [self.num_features]) # first dim is for batch
        self.batch_sample_features=tf.placeholder(tf.float32, [None] + [self.num_features]) # first dim is for batch
        self.demo_r  = tf.matmul(self.batch_demo_features,tf.expand_dims(self.phi,1))
        self.samp_r  = tf.matmul(self.batch_sample_features,tf.expand_dims(self.phi,1))
        # weights for sampled trajectories (if not using, will just be 1's)
        self.w = tf.placeholder(tf.float32, [None]+[1])
        self.irl_loss = tf.reduce_mean(self.demo_r)-tf.reduce_mean(self.w*self.samp_r)/tf.reduce_sum(self.w)
        #self.irl_loss = tf.reduce_mean(self.demo_r)-tf.reduce_mean(self.samp_r)
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(-1.0*self.irl_loss)

    def set_phi(self,phi,b=np.array([0.0])):
        '''set the parameters for ground truth simulations'''
        self.sess.run(self.phi.assign(phi))
        self.sess.run(self.b.assign(b))

    def calculate_reward(self,features):
        '''Returns reward based on a feature vector.
         Input: Features
            timesteps(or episodes) x features.
        '''
        feed_dict = {self.features:features}
        reward = self.sess.run(self.reward,feed_dict)
        return(reward)

    def update(self,
        batch_demo_features,
        batch_sample_features,
        w=None):
        '''Updates linear feature weights based on IRL loss
        Input: feature counts for demos and feature counts for samples (from optimal policy)
        '''
        # set up weights
        if w is None:
            w = np.ones((np.shape(batch_sample_features)[0],1))
        if len(w.shape)==1:
            w=w[:,np.newaxis] # add dim

        feed_dict = {self.batch_demo_features:batch_demo_features,
            self.batch_sample_features:batch_sample_features,
            self.w:w,
            }

        # perform one optimization step
        self.sess.run(self.train_op,feed_dict)

        # return loss
        return(self.sess.run(self.irl_loss,feed_dict))
