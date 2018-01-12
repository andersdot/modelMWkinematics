import imageio
import numpy as np
import sys

filenamepre = sys.argv[1]
filenamepost = sys.argv[2]
images = []
frames = np.arange(35, 165)

for i in range(9):
    for j in range(9):
        filename = '{0}_{1:03d}_{2:03d}.{3}'.format(filenamepre, i+1, j+1, filenamepost)
        print filename
        images.append(imageio.imread(filename))
imageio.mimsave(filenamepre + '.gif', images)
