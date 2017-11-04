

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import scipy.misc


class Wave1Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.viewer = None

        # need to add a reward function..

        # create grid world
        # hard coded for now
        discretize_size = 20
        time_steps = 100
        screen_dim = [780, 500]
        self.grid_size = tuple((np.array(screen_dim)/discretize_size).astype('int'))

        # other stuff
        self._seed()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size))
        # for now let's just move my guy around
        # may want to make 4D?
        # one for crystal, asteroid, alien, ship?
        # or just 1D with ship =1, crystal=2, asteroid=3, alien=4
        # if ship and crystal occupy same spot, then it's ship, and reward +=1
        # if ship and asteroid/alien occupy the same spot then game over..

        self.state = np.zeros(self.grid_size)

    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)

    def _step(self, action):

        # get current state
        state_index = list(np.where(self.state==1))
        self.state[state_index]=0 # set old ships position to 0

        # update ship's location
        if action==1:
            state_index[0] = state_index[0]+1
        if action==2:
            state_index[0] = state_index[0]-1
        if action==3:
            state_index[1] = state_index[0]+1
        if action==4:
            state_index[1] = state_index[0]-1

        # if state index > grid size or <0 keep
        # I should bounce
        for g in [0,1]:
            if state_index[g]>self.grid_size[g]:
                state_index[g]=self.grid_size[g]
            if state_index[g]<0:
                state_index[g]=0

        self.state[state_index]=1

        # update aliens location - which is just randomly taking steps

        # NEED TO IMPLEMENT

        # if run into crystal increment reward..
        reward = 0
        # NEED TO IMPLEMENT

        # if run into alien or
        end = 0
        # NEED TO IMPLEMENT

        return self.state, reward, end

    def _reset(self):
        # start in corner
        self.state = np.zeros(self.grid_size) # set all to 0 again
        self.state[0,0] = 1
        return(self.state)

    def _return_img(self):
        img = self.state
        if len(img.shape)==2:
            # add 3rd dim and repeat grid across
            img = np.repeat(img[:,:,np.newaxis],3,axis=2)
        # increase size
        img = scipy.misc.imresize(img,500)
        return(img)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        img = self._return_img()


        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
