import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

gifRes = 5

simType = "gauss"

fileName = "roe1D_" + simType + "_data.npz"

data = np.load(fileName)

q0 = data["q0"]
q1 = data["q1"]
q2 = data["q2"]
q3 = data["q3"]
q4 = data["q4"]
x = data["x"]
a = data["a"]
b = data["b"]
N = data["Nx"]
Nt = data["Nt"]
gamma = data["gamma"]
tVals = data["t"]

rho = q4
u = q1/q4
etot = q0/q4
ekin = u**2/2
eth = etot - ekin
P = (gamma-1)*rho*eth

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# fig = plt.figure(figsize=plt.figaspect(.5))
# ax1 = fig.add_subplot(2, 2, 1)
# ax2 = fig.add_subplot(2, 2, 2)
# ax3 = fig.add_subplot(2, 2, 3)
# ax4 = fig.add_subplot(2, 2, 4)
rhoplot, = ax1.plot([], [])
# uplot, = ax2.plot([], [])
# pplot, = ax3.plot([], [])
# eplot, = ax4.plot([], [])
time_text = ax1.text(0.02, 0.9, '', transform=ax1.transAxes)

def init():
    ax1.set_xlim(a, b)
    ax1.set_ylim(.75, 1.5)
    ax1.set_ylabel(r"$\rho$")

    # ax2.set_xlim(a, b)
    # ax2.set_ylim(-0.25, 2.5)
    # ax2.set_ylabel("u")

    # ax3.set_xlim(a, b)
    # ax3.set_ylim(0, 6)
    # ax3.set_ylabel("P")

    # ax4.set_xlim(a, b)
    # ax4.set_ylim(0, 30)
    # ax4.set_ylabel(r"$e_{tot}$")

    rhoplot.set_data([], [])
    # uplot.set_data([], [])
    # pplot.set_data([], [])
    # eplot.set_data([], [])
    time_text.set_text('')
    # return rhoplot, uplot, pplot, eplot, time_text
    return rhoplot, time_text

def update(frame):
    rhoplot.set_data(x, rho[:, frame*gifRes])
    # uplot.set_data(x, u[:, frame*gifRes])
    # pplot.set_data(x, P[:, frame*gifRes])
    # eplot.set_data(x, etot[:, frame*gifRes])
    
    time_text.set_text('time = %.1f' % tVals[gifRes*frame])

    # return rhoplot, uplot, pplot, eplot, time_text
    return rhoplot, time_text

anim = animation.FuncAnimation(fig, update, Nt//gifRes, init_func=init, interval=20, blit=True)

# plt.show()

f = r"c://Users/oscar/Desktop/puddle.gif"
writergif = animation.PillowWriter(fps=30)
anim.save(f, writer=writergif)

# T, X = np.meshgrid(tVals, xVals)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.plot_surface(X, T, q1Vals, cmap='inferno', linewidth=0, antialiased=False)
# ax.set_xlabel("Space")
# ax.set_ylabel("Time")
# ax.set_zlabel(r"$\q1$")

# plt.show()