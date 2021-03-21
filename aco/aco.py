# aco.py - implementation of ant colony optimization algorithm for edge detection
# Non-class stub - get the working prototype ASAP
# Source: https://github.com/hannabojadzic/ACO-for-edge-detection/blob/master/ACO-normalization/ACO_normalization.py


import cv2
import matplotlib.image as image
import numpy as np
import random
from tqdm import tqdm

ALPHA: int = 1
BETA: float = 2.0
LAMBDA: int = 10
TAU: float = 0.1    # Initial pheromone value
PHI: float = 0.05
RHO: float = 0.1
N: int = 2         # Number of iterations
L: int = 50       # Number of construction steps
K: int = 5000         # Number of ants
q_init: float = 0.8 # Controller of the degree of ant exploration
thresh: float = 0.6


def dist(x, l):
    return x * l


def normalization(img):
    A_return = np.copy(img)
    for i in range(1,len(img[1,:])-1):
        for j in range(1,len(img[:,1])-1):
            A_return[i,j] = abs(img[i-1, j-1] - img[i+1,j+1]) + abs(img[i-1,j] - img[i+1,j]) + abs(img[i-1, j+1] - img[i+1,j-1]) + abs(img[i, j-1] - img[i,j+1])
    return A_return


# Initialize image
img = image.imread('lenna_test.png')
norm_im = normalization(img)

# Create copy of input image - pheromone map
ph_map = np.full_like(img, fill_value=TAU)

# Create routes
routes = []
informations = []
for i in range(K):
    routes.append([[random.randint(1, len(img[1, :]) - 1), random.randint(1, len(img[1, :]) - 1)]])

delta_tau = []
for i in range(len(img[:,1])):
    tmp = []
    for j in range(len(img[1,:])):
        tmp.append(0.0)
    delta_tau.append(tmp)


# Populate informations list
for i in range(len(img[:,1])):
    tmp = []
    for j in range(len(img[:, 1])):
        if norm_im[i, j][0] * np.sqrt(len(norm_im[i, j])) > thresh:
            tmp.append(dist(norm_im[i,j][0], LAMBDA))
        else:
            tmp.append(0.0)
    informations.append(tmp)


# Sequential algorithm
for _ in tqdm(range(N)):
    for j in range(L):
        for k in range(K):
            i0 = routes[k][j][0]
            j0 = routes[k][j][1]

            x_min = i0 - 1
            x_max = i0 + 1
            y_min = j0 - 1
            y_max = j0 + 1

            if i0 == 0:
                x_min = 0
            if j0 == 0:
                y_min = 0

            if i0 >= len(img[1,:]) - 1:
                x_max = len(img[1,:]) - 1
            if j0 >= len(img[:,1]) - 1:
                y_max = len(img[:,1]) - 1

            neighborhood = []

            for i in range(x_min, x_max+1):
                for l in range(y_min, y_max+1):
                    if i != i0 or l != j0:
                        flag = 0
                        for pos in routes[k]:
                            if pos[0] == i and pos[1] == l:
                                flag = 1
                                break

                        if flag == 0:
                            neighborhood.append([i, l])

            u = random.random()
            p = 0

            if not neighborhood:
                m = i0
                n = j0
                routes[k].append([m, n])

            else:
                tmp = 0
                ctr = 0
                while u > p:
                    p = p + float(pow(ph_map[neighborhood[tmp][0], 
                                    neighborhood[tmp][1]][0], 
                                    ALPHA)) * float(
                                        pow(informations[neighborhood[tmp][0]][neighborhood[tmp][1]], BETA))

                    tmp += 1
                    ctr += 1
                    if tmp == len(neighborhood):
                        tmp = 0
                    
                    if ctr > 15:
                        tmp = random.randint(0, len(neighborhood))
                        break
                
                routes[k].append(neighborhood[tmp - 1])
                m = neighborhood[tmp - 1][0]
                n = neighborhood[tmp - 1][1]
            
            ph_map[m,n] = (1-PHI) * ph_map[m,n] + PHI * TAU 
            delta_tau[m][n] += informations[m][n] / float(j + 1) 

# Global update
for i in range(len(img[1,:])):
    for j in range(len(img[:,1])):
        ph_map[i,j] = (1 - RHO) * ph_map[i,j] + RHO * delta_tau[i][j]

# Generate final image
out_img = []
for i in range(0, len(img[1,:])):
    tmp = []
    for j in range(0, len(img[:,1])):
        if ph_map[i,j][0] < TAU:
            tmp.append([1.0,1.0,1.0])
        else:
            tmp.append([0.0,0.0,0.0])
            
    out_img.append(tmp)

out_img = np.array(out_img)
cv2.imshow("Output", out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()