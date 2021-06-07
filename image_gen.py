import matplotlib.pyplot as plt
import argparse
from shapes import * 
import os
import cv2 as cv

output_dir = "output/"
groundtruth_dir = output_dir + "groundtruth/"
dataset_dir = output_dir + "dataset/"

# Input arguments
parser = argparse.ArgumentParser(description='Generates predefined synthetic images for edge detection')

parser.add_argument("-f", "--figure", help="type of object to generate")
parser.add_argument("-s", "--size", help="image size", default="480x640")
parser.add_argument("-r", "--resolution", help="resolutino of figure", default="40")
parser.add_argument("-o", "--output", help="path to output image", default="output.png")
parser.add_argument("-i", "--input", help="input file with specified list of arguments")

args = parser.parse_args()
figure = args.figure
resolution = int(args.resolution)
size = args.size
(height, width) = tuple([int(f) for f in size.split("x")])
output_im = args.output
input_file = args.input


def parse_args_from_input_file(line: str) -> (str, int, int, int, int, int):
    args = line.split(" ")
    output = [None, None, None, None, None, None]

    for n, arg in enumerate(args):
        if n == 0:
            output[n] = arg
        elif n == 1:
            size = arg.split("x")[0]
            output[n] = int(size[0])
            output[n+1] = int(size[1])
        else:
            output[n+1] = int(arg)

    return tuple(output)


def prep_dirs():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(groundtruth_dir)
        os.makedirs(dataset_dir)
    elif not os.path.exists(groundtruth_dir) or not os.path.exists(dataset_dir):
        if not os.path.exists(groundtruth_dir):
            os.makedirs(groundtruth_dir)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)


def main():
    global figure, height, width, input_file, output_im, resolution
    if input_file:
        prep_dirs()
        with open(input_file, "r") as file:
            for n, line in enumerate(file.readlines()):
                try:
                    (fig, h, w, res, A, B) = parse_args_from_input_file(line)
                    if res:
                        if A and B:
                            s = Shape(res, (height, width), fig, A, B)
                        else:
                            s = Shape(res, (height, width), fig)
                    else:
                        s = Shape(resolution, (height, width), fig)
                    
                    fig_gt = s.get_figure(False, False)
                    im_gt = s.get_image(fig_gt)
                    
                    fig = s.get_figure(True, True)
                    im = s.get_image(fig)

                    s.save_image(fig_gt, groundtruth_dir+str(n)+".png")
                    s.save_image(fig, dataset_dir+str(n)+".png")
                except:
                    continue
    else:
        s = Shape(resolution, (height, width), figure)
        
        fig_gt = s.get_figure(True, False)
        im_gt = s.get_image(fig_gt)
        
        fig = s.get_figure(True, True)
        im = s.get_image(fig)

        s.save_image(fig_gt, "groundtruth_" + output_im)
        s.save_image(fig, "dataset_" + output_im)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass