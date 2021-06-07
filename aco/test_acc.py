from aco import *
from aco_accuracy import *
import cv2 as cv

gt_im = cv.imread("mandrill.png", cv.IMREAD_GRAYSCALE)

aco = ImageACO(1, 30, 1000, alpha=1.0, beta=1.0, tau=0.1,
               phi=0.05, rho=0.1, q0=0.2, image_path='mandrill.png')
aco_edge = aco()

cv.imwrite("output_im.png", aco_edge)

canny_edge = cv.Canny(cv.imread('mandrill.png'), 100, 200)
canny_edge = 255 - canny_edge

#print(f"ACO\nMSE:{MSE_index(aco_edge, gt_im):.2f}\nPSNR: {PSRN_index(aco_edge, gt_im):.2f}\n")
#print(f"Canny\nMSE: {MSE_index(canny_edge, gt_im):.2f}\nPSNR: {PSRN_index(canny_edge, gt_im):.2f}\n")

cv.imshow("Groundtruth", gt_im)
cv.imshow("Output ACO", aco_edge)
cv.imshow("Output Canny", canny_edge)
cv.waitKey(0)
