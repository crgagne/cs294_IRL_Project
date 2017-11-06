

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import scipy.misc

SHIP = np.array([1.0,0.0,0.0,0.0])
CRYSTAL =np.array([0.0,1.0,0.0,0.0])
ASTEROID = np.array([0.0,0.0,1.0,0.0])
ALIEN = np.array([0.0,0.0,0.0,1.0])

STOP = np.array([0,0])
UP = np.array([0,1])
DOWN = np.array([0,-1])
LEFT = np.array([-1,0])
RIGHT = np.array([1,0])
action_table = [STOP,UP,DOWN,LEFT,RIGHT]


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
        self.grid_size = (np.array(screen_dim)/discretize_size).astype('int')
        # print("GRID",self.grid_size)
        # other stuff
        self._seed()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size[0],self.grid_size[1],4))
        # for now let's just move my guy around
        # may want to make 4D?
        # one for crystal, asteroid, alien, ship?
        # or just 1D with ship =1, crystal=2, asteroid=3, alien=4
        # if ship and crystal occupy same spot, then it's ship, and reward +=1
        # if ship and asteroid/alien occupy the same spot then game over..


        

    def _internal_to_observation(self):
        obs = np.zeros((self.grid_size[0],self.grid_size[1],4))
        obs[self.crystal_locations.astype(np.int)] = CRYSTAL
        obs[self.ship_location.astype(np.int)] = SHIP
        obs[self.asteroid_locations.astype(np.int)] = ASTEROID
        obs[self.alien_locations.astype(np.int)] = ALIEN
        return obs


    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)

    def _step(self, action):

        # # get current state
        # state_index = list(np.where(self.state==1))
        # print(state_index[0].shape)

        # self.state[state_index]=0 # set old ships position to 0

        # update ship's location
        self.ship_location += action_table[action]
        self.ship_location = np.clip(self.ship_location,[0,0],[38,24])

        for i in range(len(self.alien_locations)):
            loc = self.alien_locations[i]
            loc += self.alien_velocities[i]
            vx,vy = self.alien_velocities[i]
            x,y = loc
            if(x < 0 or x >= self.grid_size[0]):
                vx = -vx
                if(x < 0):
                    x = 0
                else:
                    x = self.grid_size[0]-1
            elif(y < 0 or y >= self.grid_size[1]):
                vy = -vy
                if(y < 0):
                    y = 0
                else:
                    y = self.grid_size[1]-1
            self.alien_locations[i] = np.array([x,y])
            self.alien_velocities[i] = np.array([vx,vy])
            


        # print(self.alien_locations)

        # self.state[state_index]=1

        

        # update aliens location - which is just randomly taking steps

        # NEED TO IMPLEMENT
        reward = 0
        inds = (self.ship_location == self.crystal_locations).nonzero()
        before = len(self.crystal_locations) 
        self.crystal_locations = np.array([c for c in self.crystal_locations if (c != self.ship_location).any()])
        
        reward += before - len(self.crystal_locations)
        # assert len(inds) <= 1
        end = 0
        if(np.sum((self.ship_location == self.alien_locations.astype(np.int)).all(axis=-1))):
            end = 1
            # print(self.ship_location)
            # print(self.alien_locations.astype(np.int))
            # # print(np.sum())
            # print((self.ship_location == self.alien_locations.astype(np.int)).all(axis=-1))
            # print(np.sum((self.ship_location == self.alien_locations.astype(np.int)).all(axis=-1)))



        # self.state = 
        # # if(self.ship_location in self.crystal_locations):
        #     reward = 1
        #     self.crystal_locations.


        # if run into crystal increment reward..
        
        # NEED TO IMPLEMENT

        # if run into alien or
        
        # NEED TO IMPLEMENT

        return self._internal_to_observation(), reward, end

    def _reset(self):
        self.ship_location = np.array([0,0],dtype=np.int) # 1
        self.alien_locations = np.array([[38,24]],dtype=np.float) #2 
        self.alien_velocities = np.array([[-.5,-.75]],dtype=np.float) #2 
        self.crystal_locations = np.array([(1,1),(6,4),(8,9),(15,17),(16,21)],dtype=np.int) #3 
        self.asteroid_locations = np.array([(19,19),(15,22),(22,24),(30,23),(34,21)],dtype=np.int) #4
        return self._internal_to_observation()

    def _return_img(self):
        # img = self.state
        img = np.zeros((self.grid_size[0],self.grid_size[1],3))

        xs, ys = self.ship_location.transpose()

        img[xs,ys] = np.array([1.0,1.0,1.0])

        xs, ys = self.crystal_locations.transpose()
        img[xs,ys] = np.array([0.0,1.0,0.8])

        xs, ys = self.asteroid_locations.transpose()
        img[xs,ys] = np.array([1.0,0.4,0.0])

        xs, ys = self.alien_locations.astype(np.int).transpose()
        img[xs,ys] = np.array([1.0,0.0,0.0])
        img = img.transpose()
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
