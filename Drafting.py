import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd

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
"""

df = pd.read_csv("all_month.csv")
print("this is columns", df.columns)
print("this is rows", df.index)
