import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # module that gives matplotlib the ability to do 3D. Axes3D is the class that creates a 3D axis system
from matplotlib import cm # cm is colormap, a module that contains a set of predefined colormaps that can be used to map data values to colors in visualizations
import matplotlib.gridspec as gridspec #helps lay out subplots with flexible row/column sizes different from each other
import matplotlib as mpl
from matplotlib import rc, rcParams # global style setting for plots
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation



"""
Preparing the data
"""
df = pd.read_csv("all_month.csv")
# read the csv file into a pandas dataframe

df['time'] = np.sort(pd.to_datetime(df["time"], format="%Y-%m-%dT%H:%M:%S.%fZ"))
# convert the "time" column of the dataframe into datetime objects

df['time_day'] = pd.to_datetime(df['time']).dt.floor('d')
# create a new column 'time_bin' that rounds down each timestamp in the 'time'

unique_days = df['time_day'].sort_values().unique()

x1 = np.linspace(df["latitude"].min(), df["latitude"].max(), 50)
y1 = np.linspace(df["longitude"].min(), df["longitude"].max(), 50)
# creaates evenly spaced values between the min and max of latitude and longitude, with the number of values equal to the number of unique values in each column

x2, y2 = np.meshgrid(x1, y1)



"""
Create the figure and axes
"""
fig = plt.figure(000, figsize = (14, 8))
# creating the overall canvas
# 000 is an identifier for the figure, can be any number

ax = fig.add_subplot(111, projection = "3d")
# adding a 3D subplot to the figure
# can also be written as, fig.add_subplot(nrows=2, ncols=2, index=1, projection='3d')
# indexes for matplotlib is 1 based, not 0 based, as they follow the MATLAB convention



"""
Animating
"""
z_days = []
for current_day in unique_days:
    df_day = df[df['time_day'] == current_day]
    # df[df['time_day'] == current_day]
    # creates a new dataframe for each day in the unique days list (all rows where the 'time_day' column matches the current day in the loop)

    points = np.column_stack((df_day['latitude'], df_day['longitude']))
    #combine the latitude and longitude columns into a tupled array

    values = df_day['mag'].values
    # extract the magnitude values as a numpy array (df_day['mag'] also contain the index)

    if len(points) > 0:
        # to determine that it is safe (values exist) to do the interpolation

        z_day = griddata(points, values, (x2, y2), method = "linear") # Interpolate the magnitude data onto the grid (gives value, the magnitude, to the array shaped by x2 y2) using linear interpolation
        z_day_smooth = gaussian_filter(z_day, sigma=1)
        # smooth interpolation using gaussian filter
    else:
        z_day_smooth = np.zeros_like(x2) # fill empty days with zeros

    z_days.append(z_day_smooth)



smooth_frames = []
frames_per_day = 15 # number of intermediate frames between days

for i in range(len(z_days)-1):
    z_start = z_days[i]
    z_end = z_days[i+1]

    for alpha in np.linspace(0, 1, frames_per_day):
        z_interp = (z_end - z_start) * alpha + z_start
        smooth_frames.append((z_interp, unique_days[i]))
smooth_frames.append((z_days[-1], unique_days[-1]))
# ensure the last day is included



def update(frame):
        z_interp, day = smooth_frames[frame]

        for coll in ax.collections[:]:
            coll.remove()
        # clear previous surfaces

        ax.plot_surface(x2, y2, z_interp, cmap = "plasma", edgecolor='none', antialiased=True)
        ax.contourf(x2, y2, z_interp, zdir='z', offset=0, cmap=cm.plasma, antialiased=True)

        ax.set_title(f"Earthquakes by Magnitude on {day.date()}", fontsize=20, pad=20, weight='medium', y=1.02)
    

    
"""
Stylize
"""
rcParams['legend.fontsize'] = 20

rc('axes', linewidth=2) # set the axes linewidth to 2
rc('font', weight='bold', size=14) # set the font to bold and size 14

fig.suptitle("Earthquakes in the Past Month, Animated", fontsize=24, weight='bold')

ax.set_xlabel("Latitude", fontsize=10, labelpad=20, weight='medium')
ax.set_ylabel("Longitude", fontsize=10, labelpad=10, weight='medium')
ax.set_zlabel("Magnitude", fontsize=10, labelpad=10, weight='medium')

ax.set_xlim(df['latitude'].min(), df['latitude'].max())
ax.set_ylim(df['longitude'].min(), df['longitude'].max())
ax.set_zlim(0, df['mag'].max()*0.8)
# zoom in a bit on the data

ax.view_init(elev=30, azim=60)  # elev = vertical angle, azim(uth) = horizontal angle



"""
Map
"""
img = plt.imread("World-Continents-Topographic-map-Lowres.png")

img_small = img[::20, ::20]  # take every 20th pixel

lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
lat_min, lat_max = df["latitude"].min(), df["latitude"].max()

x_img = np.linspace(lon_min, lon_max, img_small.shape[1])
y_img = np.linspace(lat_min, lat_max, img_small.shape[0])
x_img, y_img = np.meshgrid(x_img, y_img)

ax.plot_surface(x_img, y_img, np.ones_like(x_img) * -0.5, rstride=1, cstride=1, facecolors=img_small/255, shade=False)

ax.set_box_aspect([abs(lat_max-lat_min), abs(lon_max-lon_min), 100])



anim = FuncAnimation(fig, update, init_func=None, frames=len(smooth_frames), interval=264, blit=False)



"""
Show
""" 
plt.show()
# shows the animated plot

interval = 200  # milliseconds between frames
fps = 1000 / interval  # frames per second
anim.save("EarthQuakes_PastMonth_Animated.mp4", writer="ffmpeg", fps=fps, dpi=300)



"""
Debug
"""
print("smooth_frames= ", smooth_frames)

print("df_day= ", df_day)
print("values= ", values)
print("df_day['mag']= ", df_day['mag'])