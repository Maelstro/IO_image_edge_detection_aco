import matplotlib.pyplot as plt
import argparse
from shapes import * 


# Input arguments
parser = argparse.ArgumentParser(description='Generates predefined synthetic images for edge detection')

parser.add_argument("-f", "--figure", help="type of object to generate", default="gaussian")
parser.add_argument("-s", "--size", help="image size", default="480x640")
parser.add_argument("-o", "--output", help="path to output image", default="output.png")

args = parser.parse_args()
figure = args.figure
size = args.size
output_im = args.output


def main():
    get_figure(figure)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass