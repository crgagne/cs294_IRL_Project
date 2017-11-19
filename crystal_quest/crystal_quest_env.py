

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import scipy.misc

SHIP = np.array([1.0,0.0,0.0,0.0])
CRYSTAL =np.array([0.0,1.0,0.0,0.0])
ASTEROID = np.array([0.0,0.0,1.0,0.0])
ALIEN = np.array([0.0,0.0,0.0,1.0])

STOP = np.array([0,0]) # 0
UP = np.array([0,-1]) # 1 
DOWN = np.array([0,1]) # 2
LEFT = np.array([-1,0]) # 3
RIGHT = np.array([1,0]) # 4
action_table = [STOP,UP,DOWN,LEFT,RIGHT]


class Wave1Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,verbose=0,num_crystals=10,num_asteroids=5):
        self.viewer = None
        self.verbose = verbose
        self.num_crystals = num_crystals
        self.num_asteroids = num_asteroids
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
        self.gate_loc = np.array([self.grid_size[0]/2,self.grid_size[1]])
        # for now let's just move my guy around
        # may want to make 4D?
        # one for crystal, asteroid, alien, ship?
        # or just 1D with ship =1, crystal=2, asteroid=3, alien=4
        # if ship and crystal occupy same spot, then it's ship, and reward +=1
        # if ship and asteroid/alien occupy the same spot then game over..

        self.min_obj_loc = (.1*self.grid_size).astype(np.int)
        self.max_obj_loc = (.9*self.grid_size).astype(np.int)


        

    def _internal_to_observation(self):
        obs = np.zeros((self.grid_size[0],self.grid_size[1],4))
        obs[self.crystal_locations.astype(np.int)] = CRYSTAL
        obs[self.ship_location.astype(np.int)] = SHIP
        obs[self.asteroid_locations.astype(np.int)] = ASTEROID
        obs[self.alien_locations.astype(np.int)] = ALIEN
        return obs


    def _seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)

    def _handle_vel(self,loc,vel):
        loc += vel#self.alien_velocities[i]
        vx,vy = vel#self.alien_velocities[i]
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

        return np.array([x,y]),np.array([vx,vy])


    def _step(self, action):

        # # get current state
        # state_index = list(np.where(self.state==1))
        # print(state_index[0].shape)

        # self.state[state_index]=0 # set old ships position to 0

        # update ship's location
        # self.ship_location += action_table[action]
        if(action != 0):
            self.ship_velocity = action_table[action]
        self.ship_location,self.ship_velocity = self._handle_vel(self.ship_location,self.ship_velocity)

        # update all alien's locations
        #TODO: Make aliens change direction randomly like in JS version

        for i in range(len(self.alien_locations)):
            loc,vel = self._handle_vel(self.alien_locations[i],self.alien_velocities[i])
            self.alien_locations[i] = loc
            self.alien_velocities[i] = vel
            if(np.random.rand() < .05):
                self.alien_velocities[i] = self._random_vel()
            

        #Check if the player has hit crystals. Allow for multple crystals in same spot.
        reward = 0
        inds = (self.ship_location == self.crystal_locations).nonzero()
        before = len(self.crystal_locations) 
        self.crystal_locations = np.array([c for c in self.crystal_locations if (c != self.ship_location).any()])
        reward += before - len(self.crystal_locations)
        

        #Check if the player has hit bad stuff or is at the gate
        end = 0
        hit_alien = np.sum((self.ship_location == self.alien_locations.astype(np.int)).all(axis=-1))
        hit_aster = np.sum((self.ship_location == self.asteroid_locations.astype(np.int)).all(axis=-1))
        at_gate = np.abs(self.ship_location[0]-self.gate_loc[0]) < 1 and self.ship_location[1] >= self.grid_size[1]-1
        if(self.verbose):
            if(hit_alien): print("HIT ALIEN")
            if(hit_aster): print("HIT ASTER")
            if(at_gate): print("AT GATE")
        cleared = len(self.crystal_locations) == 0 and at_gate
        if(hit_alien or hit_aster or cleared):
            end = 1
            

        return self._internal_to_observation(), reward, end

    def _random_points(self,low,high,num):
        xs = np.random.randint(low[0],high[0],num).reshape((num,1))
        ys = np.random.randint(low[1],high[1],num).reshape((num,1))
        return np.concatenate([xs,ys],axis=1)
    def _random_vel(self,mag=1.0):
        vel = np.random.uniform(-1,1,2)
        vel = mag*(vel/np.linalg.norm(vel))
        # xs = np.random.randint(low[0],high[0],num).reshape((num,1))
        # ys = np.random.randint(low[1],high[1],num).reshape((num,1))
        return vel
    def _reset(self):
        self.ship_location = np.array(self.grid_size/2.0,dtype=np.int) # 1
        self.alien_locations = np.array([[38,24],[0,24]],dtype=np.float) #2 
        self.alien_velocities = [self._random_vel() for _ in range(2)]
        self.crystal_locations = self._random_points(self.min_obj_loc, self.max_obj_loc,10)#np.array([(1,1),(6,4),(8,9),(15,17),(16,21)],dtype=np.int) #3 
        self.asteroid_locations = self._random_points(self.min_obj_loc, self.max_obj_loc,5) #4
        self.ship_velocity = action_table[np.random.randint(1,5)] 
        gs = self.grid_size
        portals = np.array([[0,int(gs[1]/2)-1],
                            [0,int(gs[1]/2)+0],
                            [0,int(gs[1]/2)+1],
                            [gs[0]-1,int(gs[1]/2)-1],
                            [gs[0]-1,int(gs[1]/2)+0],
                            [gs[0]-1,int(gs[1]/2)+1]],dtype=np.int) 
        self.asteroid_locations = np.concatenate([self.asteroid_locations,portals],axis=0)
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

        xs,ys = np.array([self.gate_loc+UP+2*LEFT,self.gate_loc+UP+2*RIGHT]).transpose()
        img[xs,ys] = np.array([1.0,1.0,1.0])

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
