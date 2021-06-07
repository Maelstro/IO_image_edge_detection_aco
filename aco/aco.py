# aco.py - implementation of ant colony optimization algorithm for edge detection
import sys

import cv2
import numpy as np
import random
from skimage import filters
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import copy
import os

class ImageACO(object):
    def __init__(self, iter_cnt: int, step_cnt: int, ant_cnt: int, alpha: float, beta: float,
                 tau: float, phi: float, rho: float, q0: float, image_path: str):
        self.iter_cnt = iter_cnt
        self.step_cnt = step_cnt
        self.ant_cnt = ant_cnt
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.phi = phi
        self.rho = rho
        self.q0 = q0
        self.image_path = image_path
        self.image_name = os.path.splitext(os.path.basename(self.image_path))[0]
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image = self.image.astype(np.float64)
        self.image = self.image / 255.
        self.all_visited = []
        self.img_coords = []
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                self.img_coords.append((i, j))
        self.coord_map = [i for i in range(self.image.shape[0]*self.image.shape[1])]
        self.algorithm_steps = []


        print("Initialize the pheromone map...")
        self.pheromone_map = np.full_like(self.image, fill_value=self.tau, dtype=np.float64)
        print("Pheromone map has been initialized.")

        print("Initializing the image heuristic...")
        self.heuristic_information = self.create_neighborhood()
        print("Heuristic has been initialized.")

        print("Initialize the delta_tau array...")
        self.delta_tau = np.zeros(self.image.shape, dtype=np.float64)
        print("Delta_tau array has been initialized.")

        print("Initialize routes...")
        self.routes = self.initialize_routes()
        print("Routes has been initialized.")

    def prob_per_point(self, x: float, y: float) -> float:
        result = pow(x, self.alpha) * pow(y, self.beta)
        return result

    def create_neighborhood(self) -> np.array:
        out_img = np.zeros((self.image.shape), dtype=np.float64)
        max_int = 0.
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                out_img[i, j] = abs(self.image[i-1, j-1] - self.image[i+1, j+1]) + abs(self.image[i-1, j] - self.image[i+1, j]) + \
                                abs(self.image[i-1, j+1] - self.image[i+1, j-1]) + abs(self.image[i, j-1] - self.image[i, j+1])
                if out_img[i, j] > max_int:
                    max_int = out_img[i, j]
        out_img = out_img / max_int

        return out_img

    def local_update(self, x: int, y: int) -> None:
        self.pheromone_map[x][y] = (1 - self.phi) * self.pheromone_map[x][y] + self.phi * self.tau

    def global_update(self, x: int, y: int) -> None:
        self.pheromone_map[x][y] = (1 - self.rho) * self.pheromone_map[x][y] + self.rho * self.delta_tau[x][y]

    def initialize_routes(self) -> list:
        routes = []
        for _ in range(self.ant_cnt):
            point = (random.randint(1, self.image.shape[0] - 1), random.randint(1, self.image.shape[1] - 1))
            self.all_visited.append(point)
            routes.append([point])
        return routes

    def update_route(self, ant, point):
            self.routes[ant].append(point)

    @staticmethod
    def simple_weighted_choice(choices, weights, prng=np.random):
        running_sum = np.cumsum(weights)
        u = prng.uniform(0.0, running_sum[-1])
        i = np.searchsorted(running_sum, u, side='left')
        return choices[i]

    @staticmethod
    def div(x, y):
        if y == 0:
            return 0
        else:
            return x / y

    def calculate_single_step(self, ant: int, step: int):
        # Get the pixel coordinates
        i = self.routes[ant][step][0]
        j = self.routes[ant][step][1]

        # Get the neighborhood limits
        x_min = 0 if i == 0 else i - 1
        y_min = 0 if j == 0 else j - 1

        x_max = (self.image.shape[0] - 1) if i >= self.image.shape[0] - 1 else i + 1
        y_max = (self.image.shape[1] - 1) if j >= self.image.shape[1] - 1 else j + 1

        # Iterate over neighborhood and get the probabilities
        neighborhood = []

        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                if x != i or y != j:
                    is_present = False
                    for pos in self.routes[ant]:
                        if pos[0] == x and pos[1] == y:
                            is_present = True
                            break
                    if is_present == False:
                        neighborhood.append([x, y])

        # Calculate prob per each neighborhood point
        neigh_prob = []
        for k in range(len(neighborhood)):
            neigh_prob.append(self.prob_per_point(self.pheromone_map[neighborhood[k][0], neighborhood[k][1]],
                                               self.heuristic_information[neighborhood[k][0], neighborhood[k][1]]))

        # Select a random number
        q = random.random()
        if len(neigh_prob) > 0:
            if q <= self.q0:
                m, n = neighborhood[np.argmax(neigh_prob)]
            else:
                neigh_sum = sum(neigh_prob)
                if neigh_sum == 0:
                    neigh_prob_scaled = [1 / len(neighborhood)] * len(neighborhood)
                else:
                    neigh_prob_scaled = [ImageACO.div(np_val, sum(neigh_prob)) for np_val in neigh_prob]
                idx_list = [i for i in range(len(neighborhood))]
                m, n = neighborhood[np.random.choice(idx_list, p=neigh_prob_scaled)]
        else:
            coords = self.img_coords[ImageACO.simple_weighted_choice(self.coord_map, self.heuristic_information.flatten())]
            m, n = coords

        self.update_route(ant, [m, n])


        if [m, n] not in self.all_visited:
            self.all_visited.append([m, n])

        # Locally update the pheromone map
        self.local_update(m, n)

        # Update the delta_tau
        self.delta_tau[m][n] += self.heuristic_information[m][n] / float(step + 1)

    def get_output_image(self):
        out_image = np.zeros((self.pheromone_map.shape), dtype=np.float64)
        for i in range(self.pheromone_map.shape[0]):
            for j in range(self.pheromone_map.shape[1]):
                out_image[i, j] = 0.0 if self.pheromone_map[i, j] > self.tau else 1.0

        cv2.imwrite(f"{self.image_name}_ph_map.png", out_image*255)
        return out_image

    def get_otsu_image(self):
        out_image = self.pheromone_map
        thresh = filters.threshold_otsu(out_image, nbins=256*256)
        fig = plt.figure()
        fig.canvas.set_window_title('Otsu threshold')
        plt.imshow(out_image < thresh, cmap='gray', interpolation='nearest')
        plt.show()

    def get_thresh_image(self):
        out_img = np.zeros((self.pheromone_map.shape), dtype=np.uint8)
        ph_min, ph_max = self.pheromone_map.min(), self.pheromone_map.max()
        diff = abs(ph_max - ph_min)
        for pos in self.all_visited:
            out_img[pos[0], pos[1]] = int(
                round(abs((self.pheromone_map[pos[0], pos[1]] - ph_min) / diff) * 255))

        # Invert the image
        _, out_img = cv2.threshold(out_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        cv2.imshow("Normalized threshold", out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(f"{self.image_name}_norm_thresh.png", out_img)

    def __call__(self):
        # Per iteration
        for _ in range(self.iter_cnt):
            for step in tqdm(range(self.step_cnt)):
                for ant in range(self.ant_cnt):
                    self.calculate_single_step(ant, step)
                self.algorithm_steps.append(copy.deepcopy(self.pheromone_map))
            # Global update
            for i, j in self.all_visited:
                self.global_update(i, j)

        # Call Otsu thresholding
        self.get_otsu_image()
        self.get_thresh_image()

        # Save the generated images (currently to pickle)
        with open(f"{self.image_name}_algorithm_steps.pickle", "wb") as f:
            pickle.dump(self.algorithm_steps, f)

        return self.get_output_image()

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        print("Not enough arguments! Exiting")
        sys.exit(1)
    path_to_file = args[1]
    aco = ImageACO(10, 40, 512, alpha=1.0, beta=1.0, tau=0.1, phi=0.05, rho=0.1, q0=0.6, image_path=path_to_file)
    out_edge = aco()
    cv2.imshow("Non-adaptive threshold", out_edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f"{aco.image_name}_non_adaptive.png", (out_edge*255).astype(np.uint8))