import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # module that gives matplotlib the ability to do 3D. Axes3D is the class that creates a 3D axis system
from matplotlib import cm # cm is colormap, a module that contains a set of predefined colormaps that can be used to map data values to colors in visualizations
import matplotlib.gridspec as gridspec #helps lay out subplots with flexible row/column sizes different from each other
import matplotlib as mpl
from matplotlib import rc, rcParams # global style setting for plots
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation


"""
Preparing the data
"""
df = pd.read_csv("all_month.csv")
# read the csv file into a pandas dataframe

df['time'] = np.sort(pd.to_datetime(df["time"], format="%Y-%m-%dT%H:%M:%S.%fZ"))
# convert the "time" column of the dataframe into datetime objects

df['time_bin'] = pd.to_datetime(df['time']).dt.floor('h')
# create a new column 'time_bin' that rounds down each timestamp in the 'time'

x1 = np.linspace(df["latitude"].min(), df["latitude"].max(), 50)
y1 = np.linspace(df["longitude"].min(), df["longitude"].max(), 50)
# creaates evenly spaced values between the min and max of latitude and longitude, with the number of values equal to the number of unique values in each column

x2, y2 = np.meshgrid(x1, y1)



"""
Create the figure and axes
"""
fig = plt.figure(000, figsize = (10, 6))
# creating the overall canvas
# 000 is an identifier for the figure, can be any number

ax = fig.add_subplot(111, projection = "3d")
# adding a 3D subplot to the figure
# can also be written as, fig.add_subplot(nrows=2, ncols=2, index=1, projection='3d')
# indexes for matplotlib is 1 based, not 0 based, as they follow the MATLAB convention



"""
Animating
"""
def update(frame):
    # ax.collections.clear() # Clear previous plots on the axis to avoid overplotting
    moment = df['time_bin'][frame]
    mask = df['time_bin'] == moment 
    # Create a boolean mask to filter data for the current frame based on time
    # each true/false in mask corresponds to the same row in every column! even if the column is df['latitude'], this boolean array only checks the time column

    lat = df['latitude'][mask]
    lon = df['longitude'][mask]
    mag = df['mag'][mask]
    z2 = griddata((lat, lon), mag, (x2, y2), method="cubic") # Interpolate the magnitude data onto the grid (gives value, the magnitude, to the array shaped by x2 y2) using cubic interpolation
    surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap = "viridis", edgecolor='none')
    contour = ax.contourf(x2, y2, z2, zdir='z', offset=-5, cmap=cm.viridis, antialiased=True)
    # plotting the surface on the 3D axis
    # rstride and cstride are the row and column stride

    return surf
    
anim = FuncAnimation(fig, update, init_func=None, frames=len(df['time_bin']), interval=10, blit=False)



"""
Stylize
"""

rcParams['legend.fontsize'] = 20

rc('text', usetex=True) # use LaTeX to write all text in the figure
rc('axes', linewidth=2) # set the axes linewidth to 2
rc('font', weight='bold', size=14) # set the font to bold and size 14

ax.set_title("Earthquakes in the Past Month", fontsize=24, pad=20, weight='bold')
ax.set_xlabel("Latitude", fontsize=20, labelpad=20, weight='medium')
ax.set_ylabel("Longitude", fontsize=20, labelpad=20, weight='medium')
ax.set_zlabel("Magnitude", fontsize=20, labelpad=20, weight='medium')


"""
Show
""" 
# plt.show()
# shows the animated plot

# anim.save("earthquakes.mp4", writer="ffmpeg", fps=30)


"""
Debug
"""
print(df['time_bin'][0]==df['time_bin'][1]) # checking if the first two time bins are equal, they should be because they are rounded to the nearest hour

print("this is time_bin= ", df['time_bin'])

print("time is", df['time'])
print("x1 is= ", x1)
print("latitude is= ", df["latitude"])
print("y1 is= ", y1)
print("longitude is= ", df["longitude"])

