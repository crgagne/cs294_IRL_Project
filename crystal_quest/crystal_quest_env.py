

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import scipy.misc
import random

SHIP =      np.array([1.0,0.0,0.0,0.0])*255
CRYSTAL =   np.array([0.0,1.0,0.0,0.0])*255
ASTEROID =  np.array([0.0,0.0,1.0,0.0])*255
ALIEN =     np.array([0.0,0.0,0.0,1.0])*255

NONE = np.array([0,0]) # 0
UP = np.array([0,-1]) # 1
DOWN = np.array([0,1]) # 2
LEFT = np.array([-1,0]) # 3
RIGHT = np.array([1,0]) # 4

action_table = [NONE,UP,DOWN,LEFT,RIGHT]




class Wave1Env(gym.Env):
    metadata = {'render.modes': ['human','rgb_array']}

    def __init__(self,verbose=0,
                    num_crystals=40,
                    num_asteroids=16,
                    num_aliens=1,
                    max_steps=600,
                    crystal_value=50,
                    death_value=-5,
                    screen_dim=(780,500),
                    discretize_size=20,
                    relative_window=None,
                    obs_type=0,
                    reward_func=None,
                    features=['crystal_captured',
                    'alien_collision',
                    'asteroid_collision'],
                    stochastic_actions=False,
                    choice_noise=0.1,
                    clumping_factor=.45,
                    num_crystal_clumps=15,
                    num_asteroid_clumps=5):
        self.viewer = None
        self.verbose = verbose
        self.num_crystals = num_crystals
        self.num_asteroids = num_asteroids
        self.num_aliens = num_aliens
        self.max_steps = max_steps
        self.crystal_value = crystal_value
        self.death_value = death_value
        self.obs_type = obs_type # 0 for imgs, 1 for coords
        self.reward_func=reward_func
        self.reward=0 # reward at time t
        self.clumping_factor=clumping_factor
        self.num_crystal_clumps=num_crystal_clumps
        self.num_asteroid_clumps=num_asteroid_clumps

        # useful features to pass to reward function (for timestep t)
        self.features = features
        self.crystal_captured=0
        self.asteroid_collision=0
        self.alien_collision=0
        self.dist_closest_asteroid=None
        self.dist_closest_alien=None
        # feature counts for the entire episode
        self.episode_rewards=0 # should call it episode return
        self.episode_crystals_captured=0
        self.episode_asteroid_collisions=0
        self.episode_alien_collisions=0
        self.episode_dist_closest_asteroid=0 # incrementally updated average distance to closest asteroid
        self.episode_dist_closest_alien=0

        # stochastic actions
        # one way to account for human mistakes is to assume near optimality with
        # action mistakes, where for example despite pressing up, you continue on your trajecory
        self.stochastic_actions = stochastic_actions
        self.choice_noise=choice_noise

        self.grid_size = (np.array(screen_dim)/discretize_size).astype('int')
        self.all_points = np.array([[(x,y) for x in range(self.grid_size[0])] for y in range(self.grid_size[1])],dtype=np.int).reshape(-1,2)
        self.acceptable_points = np.array([(x,y) for x,y in self.all_points if x > 0 and y > 0],dtype=np.int)
        self._seed()
        self.action_space = spaces.Discrete(5)

        # set up observation type / features
        if self.obs_type==0:
            # full image (world centric)
            self.observation_space = spaces.Box(low=0, high=1, shape=(self.grid_size[0],self.grid_size[1],4))
        elif self.obs_type==1:
            # object coordinates
            self.observation_space = spaces.Box(low=0, high=256, shape=(4+self.num_crystals*2+self.num_asteroids*2+self.num_aliens*4))
        elif self.obs_type==2:
            # distance to nearest asteroid and crystal
            self.observation_space = spaces.Box(low=0, high=1, shape=(9))
        elif self.obs_type==3:
            # relative image (relative to ship)
            assert type(relative_window) == tuple
            assert relative_window[0] % 2 == 1 and relative_window[1] % 2 == 1, "window must be odd to be symmetric"
            self.observation_space = spaces.Box(low=0, high=1, shape=(relative_window[0],relative_window[1],3))
            self.window_dist = tuple((np.array(relative_window) -1) /2)
            self.relative_window = relative_window


    def _internal_to_observation(self):

        if self.obs_type==0 or self.obs_type==3:
            # full image (world centric)
            # the uint8 is for the dqn function (saves memory)
            obs = np.zeros((self.grid_size[0],self.grid_size[1],4),dtype='uint8')
            xs, ys = self.ship_location.transpose()
            obs[xs,ys] = SHIP
            xs, ys = self.crystal_locations.transpose()
            obs[xs,ys] = CRYSTAL
            xs, ys = self.asteroid_locations.transpose()
            obs[xs,ys] = ASTEROID
            xs, ys = self.alien_locations.astype(np.int).transpose()
            obs[xs,ys] = ALIEN
        elif self.obs_type==1:
            # object coordinates
            obs = np.concatenate((self.ship_location+10.0,
                            self.ship_velocity+10.0,
                            self.crystal_locations.flatten()+10.0,
                            self.asteroid_locations.flatten()+10.0,
                            self.alien_locations[0].flatten()+10.0,
                            self.alien_velocities[0]+10.0)).astype('uint8')
        elif self.obs_type==2:
            # distance to nearest asteroid and crystal
            obs = np.zeros(9)
            closest = self.crystal_locations[np.argmin(np.sqrt(np.sum(((self.crystal_locations-self.ship_location)**2),axis=1)))]
            d_to_closest = closest-self.ship_location
            obs[0:2] = np.abs(d_to_closest)
            obs[2:4] = np.sign(d_to_closest) # this isn't ideal though because it get's transformed to 255 for -1
            closest = self.asteroid_locations[np.argmin(np.sqrt(np.sum(((self.asteroid_locations-self.ship_location)**2),axis=1)))]
            d_to_closest = closest-self.ship_location
            obs[4:6] = np.abs(d_to_closest)
            obs[6:8] = np.sign(d_to_closest)
            obs[8]=self.reward

        if self.obs_type==3:
            # relative image (relative to ship)
            x,y = self.ship_location
            dx,dy = self.window_dist
            gx, gy = self.grid_size

            # obs[x,y] = 255

            left = max(x-dx,0)
            pad_left = left - (x-dx)
            right = min(x+dx+1,gx)
            pad_right = (x+dx+1) - right

            bottom = max(y-dy,0)
            pad_bottom = bottom - (y-dy)
            top = min(y+dy+1,gy)
            pad_top = (y+dy+1) - top

            window = obs[int(left):int(right), int(bottom):int(top) ,1:]
            # print("WINDOW SLICE",window.shape)

            padtup = [(int(pad_left),int(pad_right)),(int(pad_bottom),int(pad_top)),[0,0]]
            # print(padtup)
            # print(window.shape)
            obs =np.pad(window,padtup,'constant',constant_values=255)

        # 0.0 color will be empty space.
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

        # stochastic actions, where you fail to press until next time-step
        # sets action to continue straight despite user's choice 10% of the time
        if self.stochastic_actions:
            if random.random()<self.choice_noise:
                action=0

        # update ship's location
        if(action != 0):
            self.ship_velocity = action_table[action]
        self.ship_location,self.ship_velocity = self._handle_vel(self.ship_location,self.ship_velocity)

        # update aliens' locations
        for i in range(len(self.alien_locations)):
            self.alien_locations[i],self.alien_velocities[i] = self._handle_vel(self.alien_locations[i],self.alien_velocities[i])

        # update alien velocity
        if(self.steps_taken != 0 and self.steps_taken % 5 == 0):
            self.alien_velocities = np.array([2*action_table[np.random.randint(1,5)] for _ in range(self.num_aliens)])

        # calculate distance features
        # manhatten distance to closest asteroid/alien
        self.dist_closest_asteroid = np.min([np.sum(np.abs(self.ship_location-ast_loc)) for ast_loc in self.asteroid_locations])
        self.dist_closest_alien = np.min([np.sum(np.abs(self.ship_location-al_loc)) for al_loc in self.alien_locations])
        self.episode_dist_closest_alien+=(self.dist_closest_alien-self.episode_dist_closest_alien)/(self.steps_taken+1.0) # calculate average incrementally
        self.episode_dist_closest_asteroid+=(self.dist_closest_asteroid-self.episode_dist_closest_asteroid)/(self.steps_taken+1.0)

        #Check for collisions with crystals (Update feature counts)
        inds, = (self.ship_location == self.crystal_locations).all(axis=-1).nonzero()
        if(len(inds) > 0):
            self.crystal_captured=1
            self.episode_crystals_captured+=1
            self.crystal_locations[inds] = self._random_points(len(inds),self.acceptable_points)
            print(self.crystal_locations[inds].shape,self.crystal_locations.shape)
            self.crystal_locations[inds] = self._clump(self.crystal_locations[inds],self.crystal_clumps)
        else:
            self.crystal_captured=0

        #Check for collisions with bad stuff (Update feature counts)
        self.alien_collision = np.sum((self.ship_location == self.alien_locations.astype(np.int)).all(axis=-1))
        self.episode_alien_collisions+=self.alien_collision
        self.asteroid_collision = np.sum((self.ship_location == self.asteroid_locations.astype(np.int)).all(axis=-1))
        self.episode_asteroid_collisions+=self.asteroid_collision
        assert self.asteroid_collision<2
        if(self.alien_collision or self.asteroid_collision):
            self.ship_location = np.array([19,11],dtype=np.int)#self._random_points(1,self.all_points)[0]#np.array([0,0],dtype=np.int) # 1

        #End after max_steps/10 seconds
        end = self.steps_taken >= self.max_steps

        #Tick
        self.steps_taken += 1

        # Allow for user specified reward function #
        # this is a linear reward function on features
        # could have separate reward function on obs later on
        if self.reward_func is not None:
            features = np.array([])
            for feature in self.features:
                features=np.append(features,getattr(self,feature))
            features = features[np.newaxis,:]
            self.reward = self.reward_func.calculate_reward(features)
        else:
            self.reward = -1 # baseline to prevent hiding
            self.reward += self.crystal_captured*self.crystal_value
            self.reward += self.asteroid_collision*self.death_value
            self.reward += self.alien_collision*self.death_value

        self.episode_rewards+=self.reward

        return self._internal_to_observation(), self.reward, end, {'misc info': None}


    def _random_points(self,num,select_from):
        return select_from[np.random.choice(np.arange(len(select_from)),num,replace=False)]

    def _clump(self,locations, clumps):
        closest_clumps = np.argmin(np.sum((locations.reshape(-1,1,2) - clumps.reshape(1,-1,2))**2,axis=-1),axis=1)
        print(closest_clumps,locations.shape)

        locations = (1-self.clumping_factor)*locations \
                    + self.clumping_factor*clumps[closest_clumps]
        return np.array(locations,np.int)


    def _reset(self):
        self.ship_location = np.array([19,11],dtype=np.int)#self._random_points(1,self.all_points)[0]#np.array([0,0],dtype=np.int) # 1
        if self.num_aliens==1:
            self.alien_locations = np.array([[38,24]],dtype=np.float) #2
        elif self.num_aliens==2:
            self.alien_locations = np.array([[38,24],[0,24]],dtype=np.float) #2
        self.alien_velocities = [DOWN*2 for _ in range(self.num_aliens)]#[self._random_vel() for _ in range(2)]
        random_stuff = self._random_points(self.num_crystals+self.num_asteroids,self.acceptable_points)
        self.crystal_locations = random_stuff[:self.num_crystals] #(self.min_obj_loc, self.max_obj_loc,self.num_crystals)#np.array([(1,1),(6,4),(8,9),(15,17),(16,21)],dtype=np.int) #3
        self.asteroid_locations = random_stuff[self.num_crystals:]#self._random_points(self.min_obj_loc, self.max_obj_loc,self.num_asteroids) #4
        
        #Clumping
        clumps = self._random_points(self.num_crystal_clumps+self.num_asteroid_clumps,self.acceptable_points)
        self.crystal_clumps = clumps[:self.num_crystal_clumps]
        self.asteroid_clumps = clumps[:self.num_asteroid_clumps]

        self.crystal_locations = self._clump(self.crystal_locations,self.crystal_clumps)
        self.asteroid_locations = self._clump(self.asteroid_locations,self.asteroid_clumps)
        # closest_clumps = np.argmin(np.sum((self.crystal_locations.reshape(-1,1,2) - self.crystal_clumps.reshape(1,-1,2))**2,axis=-1),axis=1)
        # print(closest_clumps.shape)
        # self.crystal_locations = (1-self.clumping_factor)*self.crystal_locations \
        #                           + self.clumping_factor*self.crystal_clumps[closest_clumps]
        # self.crystal_locations = np.array(self.crystal_locations,np.int)

        self.ship_velocity = DOWN
        gs = self.grid_size
        self.steps_taken = 0
        self.reward=0
        self.episode_rewards=0
        self.episode_alien_collisions=0
        self.episode_crystals_captured=0
        self.episode_asteroid_collisions=0
        self.episode_dist_closest_alien=0
        self.episode_dist_closest_asteroid=0

        return self._internal_to_observation()

    def _return_img(self,resize=600):
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
        img = scipy.misc.imresize(img,resize)
        return(img)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        img = self._return_img()

       # needed for gym's video recorder
        if mode == 'rgb_array':
            return img

        if mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
