import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # module that gives matplotlib the ability to do 3D. Axes3D is the class that creates a 3D axis system.
from matplotlib import cm # cm is colormap, a module that contains a set of predefined colormaps that can be used to map data values to colors in visualizations.
import matplotlib.gridspec as gridspec #helps lay out subplots with flexible row/column sizes different from each other
import matplotlib as mpl
from matplotlib import rc, rcParams # global style setting for plots
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator
import plotly.graph_objects as go

ax = plt.axes(projection = "3d")
# 3D axis system created

df = pd.read_csv("all_month.csv")
# read the csv file into a pandas dataframe

x1 = np.linspace(df["latitude"].min, df["latitude"].max(), len(df["latitude"].unique()))
y1 = np.linspace(df["longitude"].min, df["longitude"].max(), len(df["longitude"].unique()))

x2, y2 = np.meshgrid(x1, y1)

z2 = griddata((df["latitude"], df["longitude"]), df["mag"], (x2, y2), method = "cubic")
# Interpolate the magnitude data onto the grid (gives value, the magnitude, to the array shaped by x2 y2) using cubic interpolation

"""
create figure and 3d plot

plot the surface with a colormap

contours???

animate it by year

maybe style it 


x_data = df["latitude"].to_numpy()
y_data = df["longitude"].to_numpy()
z_data = df["mag"].to_numpy()
time = df["time"].to_numpy()

X, Y, Z = x_data, y_data, z_data


for i in range(0, len(time)):
    np.meshgrid(x_data[i], y_data[i], z_data[i])
    ax.plot_surface(X[i], Y[i], Z[i], cmap = "viridis")
    plt.pause(0.01)
plt.show()
"""