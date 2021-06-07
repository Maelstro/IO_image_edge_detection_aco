from aco import *
from aco_accuracy import *
import cv2 as cv

gt_im = cv.imread("../output/groundtruth/0.png", cv.IMREAD_GRAYSCALE)

cv.imshow("Groundtruth", gt_im)
cv.waitKey(0)

aco = ImageACO(10, 40, 512, alpha=1.0, beta=1.0, tau=0.1, phi=0.05, rho=0.1, q0=0., image_path='../output/dataset/0.png')
aco_edge = aco()

cv.imshow("Output ACO", aco_edge)
cv.waitKey(0)

canny_edge = cv.Canny(cv.imread('../output/dataset/0.png'), 100, 200)
canny_edge = 255 - canny_edge

cv.imshow("Output Canny", canny_edge)
cv.waitKey(0)

print(f"ACO\nMSE:{MSE_index(aco_edge, gt_im):.2f}\nPSNR: {PSRN_index(aco_edge, gt_im):.2f}\n")
print(f"Canny\nMSE: {MSE_index(canny_edge, gt_im):.2f}\nPSNR: {PSRN_index(canny_edge, gt_im):.2f}\n")