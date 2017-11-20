

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import scipy.misc

SHIP = np.array([1.0,0.0,0.0,0.0])
CRYSTAL =np.array([0.0,1.0,0.0,0.0])
ASTEROID = np.array([0.0,0.0,1.0,0.0])
ALIEN = np.array([0.0,0.0,0.0,1.0])

NONE = np.array([0,0]) # 0
UP = np.array([0,-1]) # 1
DOWN = np.array([0,1]) # 2
LEFT = np.array([-1,0]) # 3
RIGHT = np.array([1,0]) # 4

action_table = [NONE,UP,DOWN,LEFT,RIGHT]




class Wave1Env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,verbose=0,
                    num_crystals=10,
                    num_asteroids=16,
                    num_aliens=2,
                    max_steps=600,
                    crystal_value=10,
                    death_value=-100,
                    screen_dim=(780,500),
                    discretize_size=20):
        self.viewer = None
        self.verbose = verbose
        self.num_crystals = num_crystals
        self.num_asteroids = num_asteroids
        self.num_aliens = num_aliens
        self.max_steps = max_steps
        self.crystal_value = crystal_value
        self.death_value = death_value
    
        self.grid_size = (np.array(screen_dim)/discretize_size).astype('int')

        self.all_points = np.array([[(x,y) for x in range(self.grid_size[0])] for y in range(self.grid_size[1])],dtype=np.int).reshape(-1,2)
        self.acceptable_points = np.array([(x,y) for x,y in self.all_points if x > 2 and y > 2],dtype=np.int)
        print(len(self.acceptable_points))

        print(self.all_points.shape)
        self._seed()

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size[0],self.grid_size[1],4))
        # self.gate_loc = np.array([self.grid_size[0]/2,self.grid_size[1]])
        # for now let's just move my guy around
        # may want to make 4D?
        # one for crystal, asteroid, alien, ship?
        # or just 1D with ship =1, crystal=2, asteroid=3, alien=4
        # if ship and crystal occupy same spot, then it's ship, and reward +=1
        # if ship and asteroid/alien occupy the same spot then game over..

        # self.min_obj_loc = (.1*self.grid_size).astype(np.int)
        # self.max_obj_loc = (.9*self.grid_size).astype(np.int)



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
        loc += vel
        vx,vy = vel
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
        # update ship's location
        if(action != 0):
            self.ship_velocity = action_table[action]
        self.ship_location,self.ship_velocity = self._handle_vel(self.ship_location,self.ship_velocity)

        # update aliens' locations
        for i in range(len(self.alien_locations)):
            self.alien_locations[i],self.alien_velocities[i] = self._handle_vel(self.alien_locations[i],self.alien_velocities[i])


        if(self.steps_taken != 0 and self.steps_taken % 5 == 0):
            self.alien_velocities = np.array([2*action_table[np.random.randint(1,5)] for _ in range(self.num_aliens)])

        reward = 0

        #Check for collisions with crystals
        inds, = (self.ship_location == self.crystal_locations).all(axis=-1).nonzero()
        if(len(inds) > 0):
            self.crystal_locations[inds] = self._random_points(len(inds),self.acceptable_points)                
            reward += len(inds)*self.crystal_value
        
        #Check for collisions with bad stuff
        hit_alien = np.sum((self.ship_location == self.alien_locations.astype(np.int)).all(axis=-1))
        hit_aster = np.sum((self.ship_location == self.asteroid_locations.astype(np.int)).all(axis=-1))
        if(hit_alien or hit_aster):
            self.ship_location = np.array([0,0],dtype=np.int) # 1
            reward += self.death_value

        #End after max_steps/10 seconds
        end = self.steps_taken >= self.max_steps

        #Tick
        self.steps_taken += 1
        return self._internal_to_observation(), reward, end

    def _random_points(self,num,select_from):
        return select_from[np.random.choice(np.arange(len(select_from)),num,replace=False)] 
    
    def _reset(self):
        self.ship_location = np.array([0,0],dtype=np.int) # 1
        self.alien_locations = np.array([[38,24],[0,24]],dtype=np.float) #2
        self.alien_velocities = [DOWN*2 for _ in range(self.num_aliens)]#[self._random_vel() for _ in range(2)]
        random_stuff = self._random_points(self.num_crystals+self.num_asteroids,self.acceptable_points)
        self.crystal_locations = random_stuff[:self.num_crystals] #(self.min_obj_loc, self.max_obj_loc,self.num_crystals)#np.array([(1,1),(6,4),(8,9),(15,17),(16,21)],dtype=np.int) #3
        self.asteroid_locations = random_stuff[self.num_crystals:]#self._random_points(self.min_obj_loc, self.max_obj_loc,self.num_asteroids) #4
        self.ship_velocity = DOWN
        gs = self.grid_size
        self.steps_taken = 0
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
