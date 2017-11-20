import matplotlib.pyplot as plt
import numpy as np
grid = np.load('../data/tester013/testgrid.npy')
im=plt.imshow(grid[:,:,0].T,origin='upper')
for time_step in range(np.shape(grid)[2]):
    im.set_data(grid[:,:,time_step].T)
    plt.pause(0.02)
plt.show()
