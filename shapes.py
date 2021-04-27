import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import colorsys
import numpy as np
from numpy import cos, sin, sqrt, pi, cosh, sinh, exp

DPI = 300

class Shape():
    def __init__(self, resolution: int, size: (int, int), type:str, paramA=10, paramB=5):
        self.resolution = resolution                    #resolution of mesh
        (self.height, self.width) = size                #figure size
        self.type = type                                #type of shape
        self.paramA, self.paramB = paramA, paramB       #parameters used to generate shape
        self.x, self.y, self.z = self.__get_figure()    #points of shape


    def __get_figure(self):
        # Dictionary connectiong strings with functions generating shapes
        default = torus
        switcher = {
        "torus": torus,
        "sphere": sphere,
        "cone": cone,
        "gaussian": gaussian,
        "cylinder": cylinder,
        "ripple": ripple,
        "bumps": bumps,
        "shell": shell,
        "bottle": bottle,
        "breather": breather

        }
        try:
            return switcher.get(self.type)(self)
        except:
            return default(self)


    def __random_facecolors(self):
        randFaceColors = np.empty(self.x.shape, dtype=tuple)

        for y in range(len(self.y)):
            for x in range(len(self.x)):
                randHSV = (np.random.uniform(low=0.0, high=1),
                                np.random.uniform(low=0.2, high=1),
                                np.random.uniform(low=0.9, high=1))
                randFaceColors[x, y] = colorsys.hsv_to_rgb(randHSV[0], randHSV[1], randHSV[2])+(1,)

        return randFaceColors


    def get_figure(self, verbose=False, with_cmap=False):
        # Returns Image of the shape
        fig = plt.figure(dpi=DPI)
        ax = fig.gca(projection = '3d')
        fig.set_size_inches(self.width/DPI, self.height/DPI)
        if with_cmap:
            plot = ax.plot_surface(self.x, self.y, self.z, shade=False, facecolors=self.__random_facecolors(), linewidth=0)
        else:
            plot = ax.plot_surface(self.x, self.y, self.z, linewidth=0.5, color="white", shade=False, edgecolors="black")
        ax.set_zlim(-self.paramA, self.paramA)
        ax.set_xlim(np.min(self.x), np.max(self.x))
        ax.set_ylim(np.min(self.y), np.max(self.y))
        ax.axis("off")
        if verbose:
            plt.show()
        return fig
    
    def get_image(self, fig):
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()

        return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    
    def save_image(self, fig, output_file):
        fig.savefig(output_file, dpi=DPI)


def torus(shape:Shape):
    u, v = np.linspace(0, 2*pi, shape.resolution), np.linspace(0, 2*pi, shape.resolution)
    u, v = np.meshgrid(u, v)
    x = (shape.paramA + shape.paramB*cos(u))*cos(v)
    y = (shape.paramA + shape.paramB*cos(u))*sin(v)
    z = shape.paramB*sin(u)
    
    return x, y, z


def sphere(shape:Shape):
    u, v = np.linspace(-pi/2, pi/2, shape.resolution), np.linspace(0, 2*pi, shape.resolution)
    u, v = np.meshgrid(u, v)
    x = shape.paramA*cos(u)*cos(v)
    y = shape.paramA*cos(u)*sin(v)
    z = shape.paramA*sin(u)
    
    return x, y, z


def cone(shape:Shape):
    h = shape.paramA + shape.paramB
    u, v = np.linspace(0, 2*pi, shape.resolution), np.linspace(0, h, shape.resolution)
    u, v = np.meshgrid(u, v)
    x = (h-v)/h*shape.paramA*cos(u)
    y = (h-v)/h*shape.paramA*sin(u)
    z = v

    return x, y, z


def gaussian(shape:Shape):
    sigmaX = shape.paramB/2
    sigmaY = shape.paramB/2
    
    u, t = np.linspace(-shape.paramA, shape.paramA, shape.resolution), np.linspace(0, pi, shape.resolution)
    x, y = np.meshgrid(u, u)
    a = cos(t)**2/(2*sigmaX**2) + sin(t)**2/(2*sigmaY**2)
    b = -sin(2*t)/(4*sigmaX**2) + sin(2*t)/(4*sigmaY**2)
    c = sin(t)**2/(2*sigmaX**2) + cos(t)**2/(2*sigmaY**2)

    z = shape.paramA*exp( - (a*(x**2) + 2*b*x*y + c*(y**2)))
    
    return x, y, z


def cylinder(shape:Shape):
    u, v = np.linspace(0, 2*pi, shape.resolution), np.linspace(0, shape.paramA, shape.resolution)
    u, v = np.meshgrid(u, v)
    x = shape.paramA*cos(u)
    y = shape.paramA*sin(u)
    z = v
    
    return x, y, z


def ripple(shape:Shape):    
    u = np.linspace(-shape.paramA, shape.paramA, shape.resolution)
    x, y = np.meshgrid(u, u)
    z = sin((x**2+y**2)/shape.paramA)
    
    return x, y, z


def bumps(shape:Shape):
    u = np.linspace(-shape.paramA, shape.paramA, shape.resolution)
    x, y = np.meshgrid(u, u)
    z = shape.paramB*sin(x/2)*cos(y/2)/2
    
    return x, y, z


def shell(shape:Shape):
    u, v = np.linspace(0, 2*pi, shape.resolution), np.linspace(-2*pi, 2*pi, 2*shape.resolution)
    u, v = np.meshgrid(u, v)
    x = 5/4*(1 - (v/(2*pi)))*cos(2*v)*(1+cos(u))+cos(2*v)
    y = 5/4*(1 - (v/(2*pi)))*sin(2*v)*(1+cos(u))+sin(2*v)
    z = (10*v)/(2*pi)+5/4*(1 - (v/(2*pi)))*sin(u)+15
    
    return x, y, z


def bottle(shape:Shape):
    u, v = np.linspace(0, 2*pi, shape.resolution), np.linspace(0, 2*pi, shape.resolution)
    u, v = np.meshgrid(u, v)
    half = (0 <= u) & (u < pi)
    r = 4*(1 - cos(u)/2)
    x = 6*cos(u)*(1 + sin(u)) + r*cos(v + pi)
    x[half] = ((6*cos(u)*(1 + sin(u)) + r*cos(u)*cos(v))[half])
    y = 16 * sin(u)
    y[half] = (16*sin(u) + r*sin(u)*cos(v))[half]
    z = r * sin(v)
    
    return x, y, z


def breather(shape:Shape):
    a = 2/5
    u, v = np.linspace(-14, 14, 2*shape.resolution), np.linspace(-37.4, 37.4, 2*shape.resolution)
    u, v = np.meshgrid(u, v)
    x = -u +(2*(1-a**2)*cosh(a*u)*sinh(a*u))/(a*((1-a**2)*cosh(a*u)**2 + a**2*sin(sqrt(1-a**2)*v)**2))
    y = (2*sqrt(1-a**2)*cosh(a*u)*(-sqrt(1-a**2)*cos(v)*cos(sqrt(1-a**2)*v)-sin(v)*sin(sqrt(1-a**2)*v)))/(a*((1-a**2)*cosh(a*u)**2 + a**2*sin(sqrt(1-a**2)*v)**2))
    z = (2*sqrt(1-a**2)*cosh(a*u)*(-sqrt(1-a**2)*sin(v)*cos(sqrt(1-a**2)*v)+cos(v)*sin(sqrt(1-a**2)*v)))/(a*((1-a**2)*cosh(a*u)**2 + a**2*sin(sqrt(1-a**2)*v)**2))

    return x, y, z