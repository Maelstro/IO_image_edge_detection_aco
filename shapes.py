import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, sqrt, pi, cosh, sinh, exp

def torus():
    u, v = np.linspace(0, 2*pi, 40), np.linspace(0, 2*pi, 40)
    u, v = np.meshgrid(u, v)
    x = (9 + 3*cos(u))*cos(v)
    y = (9 + 3*cos(u))*sin(v)
    z = 3*sin(u)
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    plot = ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'), linewidth = 0, antialiased = False)
    ax.set_zlim(-10, 10)
    plt.show()

def sphere():
    u, v = np.linspace(-pi/2, pi/2, 40), np.linspace(0, 2*pi, 40)
    u, v = np.meshgrid(u, v)
    x = 9*cos(u)*cos(v)
    y = 9*cos(u)*sin(v)
    z = 9*sin(u)
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    plot = ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'), linewidth = 0, antialiased = False)
    ax.set_zlim(-10, 10)
    plt.show()

def cone():
    h = 10
    u, v = np.linspace(0, 2*pi, 40), np.linspace(0, h, 40)
    u, v = np.meshgrid(u, v)
    x = (h-v)/h*10*cos(u)
    y = (h-v)/h*10*sin(u)
    z = v
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    plot = ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'), linewidth = 0, antialiased = False)
    ax.set_zlim(-10, 10)
    plt.show()

def gaussian():
    sigmaX = 5
    sigmaY = 5
    
    u, v, t = np.linspace(-10, 10, 40), np.linspace(-10, 10, 40), np.linspace(0, pi, 40)
    x, y = np.meshgrid(u, v)
    a = cos(t)**2/(2*sigmaX**2) + sin(t)**2/(2*sigmaY**2)
    b = -sin(2*t)/(4*sigmaX**2) + sin(2*t)/(4*sigmaY**2)
    c = sin(t)**2/(2*sigmaX**2) + cos(t)**2/(2*sigmaY**2)

    z = 10*exp( - (a*(x**2) + 2*b*x*y + c*(y**2)))
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    plot = ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'), linewidth = 0, antialiased = False)
    ax.set_zlim(-10, 10)
    plt.show()

def cylinder():
    h = 10
    u, v = np.linspace(0, 2*pi, 40), np.linspace(0, h, 40)
    u, v = np.meshgrid(u, v)
    x = 10*cos(u)
    y = 10*sin(u)
    z = v
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    plot = ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'), linewidth = 0, antialiased = False)
    ax.set_zlim(-10, 10)
    plt.show()

def ripple():
    pass

def bumps():
    pass

def shell():
    u, v = np.linspace(0, 2*pi, 40), np.linspace(-2*pi, 2*pi, 80)
    u, v = np.meshgrid(u, v)
    x = 5/4*(1 - (v/(2*pi)))*cos(2*v)*(1+cos(u))+cos(2*v)
    y = 5/4*(1 - (v/(2*pi)))*sin(2*v)*(1+cos(u))+sin(2*v)
    z = (10*v)/(2*pi)+5/4*(1 - (v/(2*pi)))*sin(u)+15
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    plot = ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'), linewidth = 0, antialiased = False)
    plt.show()

def bottle():
    u, v = np.linspace(0, 2*pi, 40), np.linspace(0, 2*pi, 40)
    u, v = np.meshgrid(u, v)
    half = (0 <= u) & (u < pi)
    r = 4*(1 - cos(u)/2)
    x = 6*cos(u)*(1 + sin(u)) + r*cos(v + pi)
    x[half] = ((6*cos(u)*(1 + sin(u)) + r*cos(u)*cos(v))[half])
    y = 16 * sin(u)
    y[half] = (16*sin(u) + r*sin(u)*cos(v))[half]
    z = r * sin(v)
    
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    plot = ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'), linewidth = 0, antialiased = False)
    plt.show()

def breather():
    a = 2/5
    u, v = np.linspace(-14, 14, 80), np.linspace(-37.4, 37.4, 80)
    u, v = np.meshgrid(u, v)
    x = -u +(2*(1-a**2)*cosh(a*u)*sinh(a*u))/(a*((1-a**2)*cosh(a*u)**2 + a**2*sin(sqrt(1-a**2)*v)**2))
    y = (2*sqrt(1-a**2)*cosh(a*u)*(-sqrt(1-a**2)*cos(v)*cos(sqrt(1-a**2)*v)-sin(v)*sin(sqrt(1-a**2)*v)))/(a*((1-a**2)*cosh(a*u)**2 + a**2*sin(sqrt(1-a**2)*v)**2))
    z = (2*sqrt(1-a**2)*cosh(a*u)*(-sqrt(1-a**2)*sin(v)*cos(sqrt(1-a**2)*v)+cos(v)*sin(sqrt(1-a**2)*v)))/(a*((1-a**2)*cosh(a*u)**2 + a**2*sin(sqrt(1-a**2)*v)**2))

    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    plot = ax.plot_surface(x, y, z, rstride = 1, cstride = 1, cmap = plt.get_cmap('jet'), linewidth = 0, antialiased = False)
    plt.show()

def sinewave():
    pass

def get_figure(argument):
    # Dictionary connectiong strings with functions generating shapes
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
        "breather": breather,
        "sinewave": sinewave

    }
    func = switcher.get(argument)
    return func()