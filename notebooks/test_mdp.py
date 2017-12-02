import sys
sys.path.append('../crystal_quest/')
import crystal_quest_env as cq
import imp 
import numpy as np
import matplotlib.pyplot as plt
imp.reload(cq) # reload after making changes

cq.Wave1Env()

# randomly sampling actions and visualizing 
env = cq.Wave1Env(relative_window=(25,25),obs_type=3)
obs = env.reset()



for i in range(600):
    obs,r,done,_ = env.step(0 if np.random.random() < .9 else np.random.randint(1,5))#env.action_space.sample())
    if(r != 0): print("REWARD!")
    if(done != 0): print("DONE!")
    # plt.imshow(obs[:,:,:])
    # plt.show()
    env.render()
env.render(close=True)