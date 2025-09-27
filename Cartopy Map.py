import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("all_month.csv")
# read the csv file into a pandas dataframe

x1 = np.linspace(df["latitude"].min(), df["latitude"].max(), 100)
y1 = np.linspace(df["longitude"].min(), df["longitude"].max(), 100)
# creaates evenly spaced values between the min and max of latitude and longitude, with the number of values equal to the number of unique values in each column
