import numpy as np
import matplotlib.pyplot as plt

n_robs = 4
n_points_sim = [10, 30, 50, 70, 100, 200, 500, 1000]
n_sim = len(n_points_sim)

DEC_gapx_times = np.array([])
DEC_gapx_times = np.load('DEC_gapx_times.npy')
DEC_gapx_times_fast = np.load('DEC_gapx_times_fast.npy')


rob_comp_times = np.empty((n_robs, n_sim))
for i in range(n_robs):
    rob_comp_times[i] = np.load(f'rob_{i}_comp_times.npy') # Each line is a robot

print(f"DEC-gapx-GP times: {DEC_gapx_times}")

# Plot
# Mean of the computation times for each robot
rob_comp_times_mean = np.mean(rob_comp_times, axis=0)
rob_comp_times_std = np.std(rob_comp_times, axis=0)

plt.figure()
plt.plot(n_points_sim, DEC_gapx_times, label='DEC-gapx-GP', marker='o')
plt.plot(n_points_sim, rob_comp_times_mean, label='Robot means', marker='o')
plt.plot(n_points_sim, DEC_gapx_times_fast, label='DEC-gapx-GP fast', marker='o')
plt.xlabel('Robots')
plt.ylabel('Computation time (s)')
plt.title('Computation time per robot')
plt.legend()
plt.show()
