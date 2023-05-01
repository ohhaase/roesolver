import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from math import pi

gifRes = 5

simType = "freestream"

# fileName = "roe2D_" + simType + "_data.npz"

fileName = "roe2D_" + simType + "_data_cluster.npz"

data = np.load(fileName)

q0 = data["q0"]
q1 = data["q1"]
q2 = data["q2"]
q3 = data["q3"]
q4 = data["q4"]
x = data["X"]
y = data["Y"]
ax = data["ax"]
bx = data["bx"]
ay = data["ay"]
by = data["by"]
Nt = data["Nt"]
gamma = data["gamma"]
tVals = data["t"]

Vinf = 100/2
rhoinf = 1.293*2
Pinf = 101000

def trapzInt(xVals, yVals):
    return np.sum((yVals[1:] + yVals[:-1])/2 * (xVals[1:]-xVals[:-1]))

rho = q4
u = q1/q4
v = q2/q4
etot = q0/q4
ekin = (u**2 + v**2)/2
eth = etot - ekin
P = (gamma-1)*rho*eth

xmid = 0.5*(bx+ax)
ymid = 0.5*(by+ay)

r = np.sqrt((x - xmid)**2 + (y - ymid)**2)
midMask = (r < .53) & (r > .5) 

theta = np.arctan2(-(y-ymid), -(x-xmid))

surfacetheta = np.zeros(np.sum(midMask))
surfaceP = np.zeros(np.sum(midMask))

surfacetheta = theta[midMask]
sortInds = np.argsort(surfacetheta)

linTheta = np.linspace(0, 2*pi, sortInds.size)

surfPVals = [None]*tVals.size
Cd = [None]*tVals.size
Cl = [None]*tVals.size

for i in range(tVals.size):
    surfaceP = np.take_along_axis(P[i, midMask], sortInds, axis=0)
    surfPVals[i] = surfaceP
    Cd[i] = trapzInt(linTheta*.5, surfaceP*np.cos(linTheta))*2/(rhoinf*Vinf**2*2*pi)
    Cl[i] = trapzInt(linTheta*.5, surfaceP*np.sin(linTheta))*2/(rhoinf*Vinf**2*2*pi)

plt.plot(linTheta, surfaceP)

plt.show()

plt.plot(tVals, Cd)
plt.xlabel("Time, s")
plt.ylabel("$C_{d}$")

plt.show()

plt.plot(tVals, Cl)
plt.xlabel("Time, s")
plt.ylabel("$C_{l}$")

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# Pplot, = ax1.plot(linTheta, surfaceP)

# def init():
#     Pplot.set_ydata(surfaceP)
#     ax1.set_ylim(100000, 120000)

#     return Pplot,

# def animate(frame):
#     Pplot.set_ydata(surfPVals[frame*gifRes])

#     return Pplot,

# anim = animation.FuncAnimation(fig, animate, tVals.size//gifRes, init_func=init, interval=20, blit=True)



plt.show()
