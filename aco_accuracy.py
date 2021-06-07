import numpy as np
import math

def MSE_index(image, groundtruth):
    mes = np.square(np.subtrack(image, groundtruth))
    return mes

def PSRN_index(image, groundtruth):
    psnr = 10*math.log(np.max(image)**2/MSE_index(image, groundtruth))
    return psnr

def PFM_index(image, groundtruth):
    pfm = 0
    return pfm
