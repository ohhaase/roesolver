import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

gifRes = 1

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

rho = q4
u = q1/q4
v = q2/q4
etot = q0/q4
ekin = (u**2 + v**2)/2
eth = etot - ekin
P = (gamma-1)*rho*eth

# plotVals = rho

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_xlim([ax, bx])
# ax1.set_ylim([ay, by])
# ax1.set_ylabel("y")
# ax1.set_xlabel("x")
# ax1.set_aspect('equal')
# time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
# qPlot = ax1.pcolormesh(x, y, plotVals[0], cmap='inferno')

# divider = make_axes_locatable(ax1)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(qPlot, cax=cax, label=r"$\rho$")


# def init():
#     qPlot.set_array(plotVals[0])
#     time_text.set_text('')
#     qPlot.set_norm(colors.Normalize(plotVals.min(), plotVals.max()))

#     return qPlot, time_text

# def update(frame):
#     thisState = plotVals[gifRes*frame]
#     thisTime = tVals[gifRes*frame]

#     qPlot.set_array(thisState)
#     time_text.set_text('time = %.3f' % thisTime)
#     # qPlot.set_norm(colors.Normalize(thisState.min(), thisState.max()))
#     # qPlot.set_norm(colors.Normalize(1, 5))

#     return qPlot, time_text


# anim = animation.FuncAnimation(fig, update, tVals.size//gifRes, init_func=init, interval=20, blit=True)

# plt.show()


# f = r"c://Users/oscar/Desktop/density.gif"
# writergif = animation.PillowWriter(fps=30)
# anim.save(f, writer=writergif)

# plt.plot(tVals)
# plt.show()
# plt.plot(np.gradient(tVals))
# plt.show()

# thisInd = -1

# thisState = plotVals[thisInd]
# thisTime = tVals[thisInd]

# qPlot.set_norm(colors.Normalize(plotVals.min(), plotVals.max()))

# qPlot.set_array(thisState)
# time_text.set_text('time = %.3f' % thisTime)

# plt.show()

thisInd = -1

fig = plt.figure(figsize=plt.figaspect(0.75))

plotVals = [rho, P/1000, u, v]
labels = [r"$\rho \; (kg/m^3)$", "P (kPa)", "u (m/s)", "v (m/s)"]

for i in range(4):
    ax1 = fig.add_subplot(2, 2, i+1)

    dataVals = plotVals[i]

    thisState = dataVals[thisInd]
    thisTime = tVals[thisInd]

    ax1.set_xlim([ax, bx])
    ax1.set_ylim([ay, by])
    # ax1.set_ylabel("y")
    # ax1.set_xlabel("x")
    ax1.set_aspect('equal')
    if i == 0:
        time_text = ax1.text(0.02, 0.9, 'time = %.3f' % thisTime, transform=ax1.transAxes, color="w")
    qPlot = ax1.pcolormesh(x, y, thisState, cmap='inferno')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(qPlot, cax=cax, label=labels[i])

    if i == 0:
        qPlot.set_norm(colors.Normalize(dataVals.min(), dataVals.max()/2))
    else:
        qPlot.set_norm(colors.Normalize(dataVals.min(), dataVals.max()))

plt.show()