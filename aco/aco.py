# aco.py - implementation of ant colony optimization algorithm for edge detection
# Non-class stub - get the working prototype ASAP
# Source: https://github.com/hannabojadzic/ACO-for-edge-detection/blob/master/ACO-normalization/ACO_normalization.py


import cv2
import matplotlib.image as image
import numpy as np
import random
from tqdm import tqdm

class ImageACO(object):
    def __init__(self, iter_cnt: int, step_cnt: int, ant_cnt, alpha: int, beta: float,
                 tau: float, phi: float, rho: float, image_path: str):
        self.iter_cnt = iter_cnt
        self.step_cnt = step_cnt
        self.ant_cnt = ant_cnt
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.phi = phi
        self.rho = rho
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        print("Initialize the pheromone map...")
        self.pheromone_map = np.full_like(self.image, fill_value=self.tau)
        print("Pheromone map has been initialized.")

        print("Initializing the image heuristic...")
        self.heuristic_information = self.create_neighborhood()
        print("Heuristic has been initialized.")

        print("Initialize the delta_tau array...")
        self.delta_tau = np.zeros(self.image.shape)
        print("Delta_tau array has been initialized.")

        print("Initialize routes...")
        self.routes = self.initialize_routes()
        print("Routes has been initialized.")

    def prob_per_point(self, x: float, y: float) -> float:
        result = pow(x, self.alpha) * pow(y, self.beta)
        return result

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

    def initialize_routes(self) -> list:
        routes = []
        for _ in range(self.ant_cnt):
            routes.append([[random.randint(1, self.image.shape[0] - 1), random.randint(1, self.image.shape[1] - 1)]])
        return routes

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
        max_prob = 0.0
        prob_x = 0
        prob_y = 0

        for x in range(x_min, x_max+1):
            for y in range(y_min, y_max+1):
                tmp_prob = self.prob_per_point(self.image[x][y], self.heuristic_information[x][y])
                if tmp_prob >= max_prob:
                    max_prob = tmp_prob
                    prob_x = x
                    prob_y = y

        # Update the route with the most probable neighbor
        self.routes[ant].append([prob_x, prob_y])

        # Locally update the pheromone map
        self.local_update(i, j)

        # Update the delta_tau
        self.delta_tau[i][j] += self.heuristic_information[i][j]

    def get_output_image(self):
        ret, out_image = cv2.threshold(self.pheromone_map, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return out_image

    def __call__(self):
        # TODO: Finish the method for edge detection
        # TODO: Calculate probability, add the neighbor with the biggest prob to the route
        # Per iteration
        for _ in range(self.iter_cnt):
            for step in range(self.step_cnt):
                for ant in range(self.ant_cnt):
                    self.calculate_single_step(ant, step)

            # Global update
            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    self.global_update(i, j)

        return self.get_output_image()

if __name__ == "__main__":
    aco = ImageACO(2, 50, 5000, alpha=1, beta=2.0, tau=0.1, phi=0.05, rho=0.1, image_path='lenna_test.png')
    out_edge = aco()
    cv2.imshow("Output", np.asarray(out_edge))
    cv2.waitKey(0)
    cv2.destroyAllWindows()