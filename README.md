3D meshgrid displaying the location (latitude, longitude) and intensity of earthquakes across the world in the past month, between the dates 8-19-2025 and 9-18-2025.

TO BE IMPLEMENTED:
- map on the bottom to display where in the world the earthquakes occur
- faster animated video


!!!important text displayed in this format!!!

___
Long animation settings
___
<video controls src="EarthQuakes_PastMonth_Animated_Long.mp4" title="Title"></video>
In EarthQuakes_PastMonth_Animated_Areas.py
Line 128:
anim = FuncAnimation(fig, update, init_func=None, frames=len(smooth_frames), !!!interval=264!!!, blit=False)
line 140:
interval = 200  # milliseconds between frames
fps = 1000 / interval  # frames per second
anim.save("EarthQuakes_PastMonth_Animated.mp4", writer="ffmpeg", !!!fps=fps!!!, dpi=300)