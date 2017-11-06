import sys
sys.path.append('../crystal_quest/')
import crystal_quest_env as cq
import imp 
import numpy as np
imp.reload(cq) # reload after making changes

cq.Wave1Env()

# randomly sampling actions and visualizing 
env = cq.Wave1Env(verbose=1)
obs = env.reset()



for i in range(500):
    obs,r,done = env.step(np.random.randint(1,5) if np.random.random() < .2 else 0)
    if(r != 0): print("REWARD!")
    if(done != 0): print("DONE!")
    env.render()
env.render(close=True)