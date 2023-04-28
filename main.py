import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Dimensions and step
plate_length = 50
max_iter_time = 5000
delta_x = 1

# Material properties - copper
c = 390 #J/kg*K
rho = 8960 #kg/m^3
lambda_xyz = 385 #W/(m*K)
alpha = lambda_xyz / (rho * c)

# Time Parameters
delta_t = 0.001
gamma = (alpha * delta_t) / ((delta_x/1000) ** 2)

# Initialize solution on the grid of T(k, i, j)
T = np.empty((max_iter_time, plate_length, plate_length))

# Initial condition everywhere inside the grid
T_initial = 20.0

# Boundary conditions
T_top = 100.0
T_left = 20.0
T_bottom = 20.0
T_right = 20.0

# Initial condition
T.fill(T_initial)

# Boundary conditions
T[:, (plate_length-1):, :] = T_top
T[:, :, :1] = T_left
T[:, :1, 1:] = T_bottom
T[:, :, (plate_length-1):] = T_right

def calculate(u):
    for k in range(0, max_iter_time-1, 1):
        for i in range(1, plate_length-1, delta_x):
            for j in range(1, plate_length-1, delta_x):
                u[k + 1, i, j] = gamma * (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j]) + u[k][i][j]

    return u

def plotheatmap(u_k, k):
    # Clear the current plot figure
    plt.clf()

    plt.title(f"Temperature [deg C] at t = {k*delta_t:.2f} s for copper")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")

    # This is to plot u_k (u at time-step k)
    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=20, vmax=100)
    plt.colorbar()

    return plt

# Temperature calculation here
T = calculate(T)

def animate(k):
    plotheatmap(T[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=max_iter_time, repeat=False)
anim.save("temperature_distribution_copper.gif")

# Saving some distributions at given time
for k in range(250,2000,250):
    plot=plotheatmap(T[k],k)
    plt.savefig(f"temp_dist_cop_{k/250:.0f}")

# Calculation of time to achieve steady-state
converged=False
epsilon=0.001
k=0
while not converged and k <= max_iter_time-1:
    if np.max(np.abs(T[k + 1] - T[k])) < epsilon:
        converged = True
    k += 1
if converged:
    print(f"Steady state reached at t = {k*delta_t:.2f} s")
else:
    print("Steady state not reached")