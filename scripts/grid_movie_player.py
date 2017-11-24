import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob

'''Example: python grid_movie_player.py --folder ../data/A1GXFMAC759VRM --file_num 6'''

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='../data/tester013')
    parser.add_argument('--file_num', type=int, default=0)
    args = parser.parse_args()

    files=glob.glob(args.folder+'/*.npy')
    print(files)

    grid = np.load(files[args.file_num])

    im=plt.imshow(grid[:,:,0].T,origin='upper')
    for time_step in range(np.shape(grid)[2]):
        im.set_data(grid[:,:,time_step].T)
        plt.pause(0.01)
    plt.show()

if __name__ == "__main__":
    main()
