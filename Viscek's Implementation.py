#Below is the implementation of Vicseks Model

import numpy as np
import matplotlib.pyplot as plt


def vicsek(N, eta, v0, r, L, T):                                                #Define a function to implement Vicseks Model
    init_pos = L * np.random.rand(N, 2)                                         #Initialize Initial Position
    pos = init_pos.copy()                                                       #Initialize Final Position
    vel = v0 * np.random.randn(N, 2)                                            #Initialize Velocity
    vel /= np.linalg.norm(vel, axis=1)[:, np.newaxis]                           #Normalize velocities

    
    num_steps = int(T / r)                                                      #Iterative cycle for Simulating Vicsek's model
    for step in range(num_steps):
        
        dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)   #Compute Neighbourhood
        mask = (dists > 0) & (dists < eta)
        #neighbors = np.array([np.where(mask[i])[0] for i in range(N)])
        neighbors = [list(np.where(mask[i])[0]) for i in range(N)]

        
        mean_dir = np.array([np.mean(vel[neighbors[i]], axis=0) for i in range(N)])     #Compute mean direction
        mean_dir /= np.linalg.norm(mean_dir, axis=1)[:, np.newaxis]

    
        vel += r * mean_dir                                                         #Update Velocity
        vel /= np.linalg.norm(vel, axis=1)[:, np.newaxis]                           #Normalize velocities
        pos += r * vel                                                              #Update Position

    
        pos[pos < 0] += L                       #Bringing the position back in the range of [0,L] if position less than 0
        pos[pos > L] -= L                       #Bringing the position back in the range of [0,L] if position more than L

    return init_pos, pos


N1 = 100                        #Number of particles for Simulation 1                         
N2 = 500                        #Number of particles for simulation 2
eta = 2                         #Noise
v0 = 0.1                        #Initial Velocity
r = 0.1                         #Time step
L = 10.0                        #Length of Sqaure         
T = 10.0                        #Total time taken for simulation

init_pos1, pos1 = vicsek(N1, eta, v0, r, L, T)              #Running the simulation with N1 particles
init_pos2, pos2 = vicsek(N2, eta, v0, r, L, T)              #Running the simulation with N2 particles

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))       #Plotting Initial and Final state of N=100 particles
ax1.scatter(init_pos1[:, 0], init_pos1[:, 1], s=10, marker='1', color='orange', label='Particles at Initial State')
ax2.scatter(pos1[:, 0], pos1[:, 1], s=10, marker='1', color='green', label='Particles at Final State')
ax1.set_title(f"N = {N1} Initial state")
ax1.legend(loc='upper right')
ax2.set_title(f"N = {N1} Final state")
ax2.legend(loc='upper right')
plt.show()

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(10, 5))       #Plotting Initial and Final state of N=500 particles  
ax3.scatter(init_pos2[:, 0], init_pos2[:, 1], s=10, marker='1', color='orange', label='Particles at Initial State')
ax4.scatter(pos2[:, 0], pos2[:, 1], s=10, marker='1', color='green', label='Particles at Final State')
ax3.set_title(f"N = {N2} Initial state")
ax3.legend(loc='upper right')
ax4.set_title(f"N = {N2} Final state")
ax4.legend(loc='upper right')
plt.show()
