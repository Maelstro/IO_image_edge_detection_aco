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
N: int = 2          # Number of iterations
L: int = 50         # Number of construction steps
K: int = 5000       # Number of ants
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
    routes.append([[random.randint(1, len(img[1, :]) - 1), random.randint(1, len(img[:, 1]) - 1)]])

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

##### Class refactoring #####

class ImageACO(object):
    def __init__(self, iter_cnt: int, step_cnt: int, ant_cnt, alpha: int, beta: float,
                 lambda_val: float, tau: float, phi: float, rho: float, image_path: str):
        self.iter_cnt = iter_cnt
        self.step_cnt = step_cnt
        self.ant_cnt = ant_cnt
        self.alpha = alpha
        self.beta = beta
        self.lambda_val = lambda_val
        self.tau = tau
        self.phi = phi
        self.rho = rho
        self.pheromone_map = None
        self.delta_tau = None
        self.heuristic_information = None
        self.routes = None
        self.image = cv2.imread(image_path)

    def __enter__(self):
        print("Initialize the pheromone map...")
        self.pheromone_map = np.full_like(self.image, fill_value=self.tau)
        print("Pheromone map has been initialized.")

        print("Initializing the image heuristic...")
        self.heuristic_information = self.create_neighborhood()
        print("Heuristic has been initialized.")

        print("Initialize the delta_tau array...")
        self.delta_tau = np.zeros(self.image)
        print("Delta_tau array has been initialized.")

    def prob_per_point(self, x: int, y: int) -> float:
        return pow(x, self.alpha) * pow(y, self.beta)

    def create_neighborhood(self) -> np.array:
        out_img = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.float32)
        max_int = 0.
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                out_img[i][j] = abs(self.image[i-1][j-1] - self.image[i+1][j+1]) + abs(self.image[i-1][j] - self.image[i+1][j]) + \
                                abs(self.image[i-1][j+1] - self.image[i+1][j-1]) + abs(self.image[i][j-1] - self.image[i][j+1])
                if out_img[i][j] > max_int:
                    max_int = out_img[i][j]
        out_img = out_img / max_int
        return out_img

    def local_update(self, x: int, y: int) -> None:
        self.pheromone_map[x][y] = (1 - self.phi) * self.pheromone_map[x][y] + self.phi * self.tau

    def global_update(self, x: int, y: int) -> None:
        self.pheromone_map[x][y] = (1 - self.phi) * self.pheromone_map[x][y] + self.phi * self.delta_tau[x][y]

    def initialize_routes(self):
        self.routes = []
        for _ in range(self.ant_cnt):
            self.routes.append([[random.randint(1, self.image.shape[0] - 1), random.randint(1, self.image.shape[1] - 1)]])

    def __call__(self):
        # TODO: Finish the method for edge detection
        # TODO: Calculate probability, add the neighbor with the biggest prob to the route
        pass