import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # module that gives matplotlib the ability to do 3D. Axes3D is the class that creates a 3D axis system
from matplotlib import cm # cm is colormap, a module that contains a set of predefined colormaps that can be used to map data values to colors in visualizations
import matplotlib.gridspec as gridspec #helps lay out subplots with flexible row/column sizes different from each other
import matplotlib as mpl
from matplotlib import rc, rcParams # global style setting for plots
import pandas as pdpip
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
import random as rd
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
import matplotlib.image as mpimg
from matplotlib.cbook import get_sample_data
from skimage.transform import resize





"""
ar = np.array([1, 2, 3, 4, 5, 6])
print(ar)

ar2d = np.array([[1, 2, 3], [4, 5, 6]])
print(ar2d)

zeros = np.zeros(5)
print(zeros)

ones = np.ones((3, 4))


headTails = [0, 0]

for _ in range(100000):
    headTails[rd.randint(0, 1)] += 1
    plt.bar(["Heads", "Tails"], headTails, color = ["blue", "orange"])
    plt.pause(0.001)
plot.show()

"""

"""
ax = plt.axes(projection = "3d")

x_data = np.arange(-5, 5, 1)
y_data = np.arange(-5, 5, 1)

for i in range(1, len(x_data)):
    X, Y = np.meshgrid(x_data[:i], y_data[:i])
    print("X is= ", X)
    print("Y is= ", Y)
    X, Y = np.meshgrid(x_data, y_data)
    Z = np.sin(X) * np.cos(Y) * rd.uniform(0, 100)
    print("Z is= ", Z)
    print("RANDOM (not the one in the program) rd.uniform is= ", rd.uniform(0, 10))
    ax.plot_surface(X, Y, Z, cmap = "viridis")
    plt.pause(1)
    print("paused")

plt.show()
print("Done")
"""

"""
df = pd.read_csv("Salary_Data.csv")
years = df["YearsExperience"].to_numpy()
salary = df["Salary"].to_numpy()

earthQuake = pd.read_csv("all_month.csv")
print(earthQuake)
"""

"""
a = np.array([2.5, 2, 3, 5, 4, 5])
print(np.unique(np.sort(a)))


df = pd.read_csv("all_month.csv")
print("this is columns", df.columns)
print("this is rows", df.index)
"""


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# set up figure

fn = get_sample_data("E:\Github Projects\DataVisualization_5913\World-Continents-Topographic-map-Lowres.png", asfileobj=False)
# might need os to set the path later on if this has to be replicable

arr = mpimg.imread(fn)
ny, nx, _ = arr.shape  # image height, width

if arr.shape[2] == 4: arr = arr[:, :, :3]  # RGB only

downscale = 10   # increase for faster render, decrease for more detail
nx_ds, ny_ds = nx // downscale, ny // downscale
# cuz it keeps crashing

x = np.linspace(-5, 5, nx_ds)
y = np.linspace(-5, 5, ny_ds)
X1, Y1 = np.meshgrid(x, y)
# Create meshgrid for sine surface

R = np.sqrt(X1**2 + Y1**2)
Z = np.sin(R)

surf = ax.plot_surface(X1, Y1, Z, rstride=1, cstride=1, cmap=cm.winter, linewidth=0, antialiased=True)
# plot sine surface

ax.set_zlim(-2.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# NoteToSelf heck does this do lol check it out later

"""
# 10 is equal length of x and y axises of your surface
stepX, stepY = 10. / arr.shape[0], 10. / arr.shape[1]
X1 = np.arange(-5, 5, stepX)
Y1 = np.arange(-5, 5, stepY)
X1, Y1 = np.meshgrid(X1, Y1)
# stride args allows to determine image quality 
# stride = 1 work slow
"""

arr_resized = resize(arr, (ny_ds, nx_ds, 3), anti_aliasing=True)
# Reshape to match X1/Y1 (they expect 2D)

imgZ = np.full_like(X1, -2.25)


"""
min_z = Z.min()  # usually about -1
imgZ = np.full_like(X1, min_z - 0.1)  # put it a bit lower
# min Z
"""

facecolors = arr_resized / 255.0 if arr.max() > 1 else arr_resized
# place image as flat plane at z=0

step = 5
ax.plot_surface(X1[::step, ::step], Y1[::step, ::step], imgZ[::step, ::step], rstride=1, cstride=1, facecolors=facecolors[::step, ::step, :], linewidth=0, antialiased=False)


"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X = np.arange(-5, 5, .25)
Y = np.arange(-5, 5, .25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.winter,
                       linewidth=0, antialiased=True)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fn = get_sample_data("E:\Github Projects\DataVisualization_5913\World-Continents-Topographic-map-Lowres.png", asfileobj=False)
arr = mpimg.imread(fn)
# 10 is equal length of x and y axises of your surface
stepX, stepY = 10. / arr.shape[0], 10. / arr.shape[1]

X1 = np.arange(-5, 5, stepX)
Y1 = np.arange(-5, 5, stepY)
X1, Y1 = np.meshgrid(X1, Y1)
# stride args allows to determine image quality 
# stride = 1 work slow

imgZ = np.full_like(X1, 0)
ax.plot_surface(X1, Y1, imgZ, rstride=1, cstride=1, facecolors=arr)

"""
plt.show()
fig.savefig("Map Testing Draft.png", dpi=300)
