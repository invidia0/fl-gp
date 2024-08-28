import numpy as np
import utilities as utils
import matplotlib.pyplot as plt
from robot import Robot
import time
import os
import glob
import concurrent.futures
from scipy.interpolate import Rbf

np.random.seed(0)

############################################################################################################
""" Dataset (ROMS, Intel Berkeley) """
data_file = 'data_filtered-time.txt'
positions_file = 'positions.txt'

values = np.empty((54, 1440))

with open(data_file, "r") as file:
    for i, line in enumerate(file):
        data = line.split()
        values[i] = data

positions = np.empty((54, 2))
with open(positions_file, "r") as file:
    for i, line in enumerate(file):
        data = line.split()
        positions[i] = data[1:3]
############################################################################################################

# x_inf, y_inf = 0, 0
# x_sup, y_sup = 42, 31
# BBOX = [x_inf, y_inf, x_sup, y_sup]
# delta = 1

# # Create the field of the environment with measurements and the interpolation
# x_locs = positions[:,0]
# y_locs = positions[:,1]

# # Values are referred to the top-right corner of the environment, change to the bottom-left corner
# y = y_sup - x_locs.astype(float)
# x = x_sup - y_locs.astype(float)

# # Create the grid
# x_vals = np.arange(x_inf, x_sup + delta, delta)
# y_vals = np.arange(y_inf, y_sup + delta, delta)
# x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
# X12MESH = np.c_[x_mesh.ravel(), y_mesh.ravel()]

# Generate data
np.random.seed(0)

area_size = 40
x_inf, y_inf = 0, 0
x_sup, y_sup = area_size, area_size
BBOX = [x_inf, y_inf, x_sup, y_sup]
d_field_ = 1
x1_ = np.arange(0, area_size + d_field_, d_field_)
x2_ = np.arange(0, area_size + d_field_, d_field_)
_X1, _X2 = np.meshgrid(x1_, x2_)
mesh = np.vstack([_X1.ravel(), _X2.ravel()]).T

# Generate random means
peaks = 8 # np.random.randint(1, 10)
means = np.random.uniform(low=0, high=area_size, size=(peaks, 2))
sigma = 5
Z = utils.gmm_pdf_array(mesh[:, 0], mesh[:, 1], sigma, means, flag_normalize=False)
Z = Z.reshape(len(x1_), len(x2_))

""" Robots parameters """
ROB_NUM = 4
CAMERA_BOX = 2

_area_to_cover = (x_sup * y_sup) * 1.0

RANGE = np.sqrt((_area_to_cover / ROB_NUM) / np.pi) * 2

K_GAIN = 3
D_t = 0.1

fig, axes = plt.subplots(4, 2)
# set axes aspect ratio to be equal
for ax in axes.flatten():
    ax.set_aspect('equal')

robots = np.empty(ROB_NUM, dtype=object)

safety_dist = 0.5
for r in np.arange(ROB_NUM):
    # if r == ROB_NUM - 1:
    #     x1, x2 = x_sup - safety_dist, y_sup - safety_dist
    # else:
    x1, x2 = np.random.uniform(0 + 10, (x_sup - safety_dist)/2), np.random.uniform(0 + safety_dist,( y_sup - safety_dist)/2)
    rob = Robot(total_robots=ROB_NUM,
                id=r,
                x1_init=x1,
                x2_init=x2,
                sensing_range=RANGE,
                sensor_noise=0.05,
                bbox=BBOX,
                mesh=mesh,
                field_delta=d_field_)
    robots[r] = rob

""" Figures """
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
plt.rcParams["pdf.fonttype"] = 42

PERIOD = 301
index_init = 600
index_end = 1400
step = (index_end - index_init) // PERIOD
index = index_init

A = np.zeros((ROB_NUM, ROB_NUM)) # Adjacency matrix

""" DEC-apx-GP """
s_end_DEC_gapx = 100
rho = 500
ki = 5000
TOL_ADMM = 1e-3

""" DEC-PoE """
beta = 1 / ROB_NUM
s_end_DAC = 1000

"""
Main loop
"""
for t in np.arange(0, PERIOD):
    print(f"\n*** Step: {t} ***")
    # W_t = np.tanh(1 * t)
    W_t = 10

    if t == 0:
        first = True
    else:
        first = False

    # temp = values[:, index]
    # temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
    # f_eval = Rbf(x, y, temp, function='thin_plate', smooth=0.0)
    # FIELD = f_eval(x_mesh, y_mesh)
    
    utils.sense_neighbors(robots)
    utils.update_A(robots, A)
    utils.find_groups(robots, A)
    groups = [robot.group for robot in robots]

    degrees = np.sum(A, axis=1)
    max_degree = np.max(degrees)

    eps = 1 / (max_degree)

    for i, robot in enumerate(robots):
        # Set the robot's group
        robot.set_Wt = W_t
        # robot.group = groups[i]
        # robot.set_field(FIELD)
        robot.compute_voronoi()
        
        # if SPARSE:
        # vertices = robot.diagram
        # vertXinf, vertYinf = np.min(vertices, axis=0)
        # vertXsup, vertYsup = np.max(vertices, axis=0)
        
        # x_left, x_right = robot.position[0] - CAMERA_BOX, robot.position[0] + CAMERA_BOX
        # y_bottom, y_top = robot.position[1] - CAMERA_BOX, robot.position[1] + CAMERA_BOX
        # # Check that the camera box is inside the voronoi diagram
        # if x_left < vertXinf:
        #     x_left = vertXinf
        # if x_right > vertXsup:
        #     x_right = vertXsup
        # if y_bottom < vertYinf:
        #     y_bottom = vertYinf
        # if y_top > vertYsup:
        #     y_top = vertYsup

        # num_points_x = 5 # Adjust the number of points as needed
        # num_points_y = 5 # Adjust the number of points as needed

        # x_vals = np.linspace(x_left, x_right, num_points_x)
        # y_vals = np.linspace(y_bottom, y_top, num_points_y)

        # x_vals, y_vals = np.meshgrid(x_vals, y_vals)
        # points = np.stack([x_vals.ravel(), y_vals.ravel()], axis=1)
        # y_values = field_eval(points[:, 0], points[:, 1]) + robot.sensor_noise * np.random.randn(len(points))

        # Take 5 random points from the robot sensing area
        points = np.random.uniform(robot.position - CAMERA_BOX, robot.position + CAMERA_BOX, (5, 2))
        # y_values = f_eval(points[:, 0], points[:, 1]) + robot.sensor_noise * np.random.randn(len(points))
        y_values = utils.gmm_pdf_array(points[:, 0], points[:, 1], sigma, means, flag_normalize=False) + robot.sensor_noise * np.random.randn(len(points))
        
        # y_value = f_eval(robot.position[0], robot.position[1]) + np.random.normal(0, robot.sensor_noise**2)
        robot.sense(points, y_values, first=first)
        robot.update_dataset() # Update the dataset with the new observation

    # for robot in robots:
    #     for other_robot in robot.neighbors:
    #         robot.sense(other_robot.observations[:, :2], other_robot.observations[:, 2], first=first)

    for robot in robots:
        robot.update_dataset()
        print(f"Robot {robot.id} has {robot.observations.shape[0]} observations")

    for robot in robots:
        robot.filter_dataset()
        print(f"Robot {robot.id} has {robot.dataset.shape[0]} filtered observations")

    for group in np.unique(groups):
        print(f"Group {group} is being processed")
        robots_group = [robot for robot in robots if robot.group == group]
        old_hypers = np.zeros((len(robots_group), 3)) # Initialize hypers with zeros

        for s in range(s_end_DEC_gapx):
            tmp_hyps = np.empty((0, 3)) # Temporary store the new hyperparameters
            for id, robot in enumerate(robots_group):
                # Take the neighbors' hyperparameters
                neigbors_hyps = np.empty((0, 3))
                for neighbor in robot.neighbors:
                    neigbors_hyps = np.vstack((neigbors_hyps, neighbor.hyps))
                n_neighbors = len(neigbors_hyps)    

                # Duals (30a) (Consensus)
                sum = np.array([0, 0, 0])
                for k in range(n_neighbors):
                    sum = sum + (robot.hyps - neigbors_hyps[k])
                robot.p = robot.p + rho * sum

                # Primal (34b) (ADMM)
                first_term = rho * np.sum(neigbors_hyps, axis=0)
                second_term = utils.log_likelihood_grad(robot)
                third_term = (ki + n_neighbors * rho) * robot.hyps
                res = (1 / (ki + 2 * n_neighbors * rho)) * (first_term - second_term + third_term - robot.p)
                tmp_hyps = np.vstack([tmp_hyps, res])

            old_hypers = np.copy([robot.hyps for robot in robots_group])
            for i, robot in enumerate(robots_group):
                robot.hyps = tmp_hyps[i]

        for robot in robots_group:
            print(f"Robot {robot.id} has hyperparameters: {robot.hyps}")
        print(f"- Done! - ")

        """ DEC-PoE """
        print(f"DEC-PoE for group {group}")
        # Compute the local predictions
        for robot in robots_group:
            robot.update_estimate()

        # Initialize the weights
        for robot in robots_group:
            robot.w_mu = beta * robot.cov_rec * robot.mean
            robot.w_cov = beta * robot.cov_rec
                    
        for s in range(s_end_DAC):
            for robot in robots_group:
                neighbors_w_mu = []
                neighbors_w_cov = []
                for other_robot in robot.neighbors:
                    neighbors_w_mu.append(other_robot.w_mu)
                    neighbors_w_cov.append(other_robot.w_cov)
                n_neighbors = len(neighbors_w_mu)

                # DAC 1 (Mean)
                sum = np.zeros([len(x1_), len(x2_)], dtype=np.float128)
                for k in range(n_neighbors):
                    sum = sum + (neighbors_w_mu[k] - robot.w_mu)
                robot.tmp_w_mu = robot.w_mu + eps * sum

                # DAC 2 (coviance)
                sum = np.zeros([len(x1_), len(x2_)], dtype=np.float128)
                for k in range(n_neighbors):
                    sum = sum + (neighbors_w_cov[k] - robot.w_cov)
                robot.tmp_w_cov = robot.w_cov + eps * sum

            for robot in robots_group:
                robot.w_mu = robot.tmp_w_mu
                robot.w_cov = robot.tmp_w_cov

            for robot in robots_group:
                robot.cov_rec = ROB_NUM * robot._w_cov
                robot.cov_rec[robot.cov_rec < 0] = 0
                np.nan_to_num(robot.cov_rec, copy=False)
                # Update the std and cov
                robot.std = np.sqrt(1/robot.cov_rec)
                robot.mean = (1 / robot.cov_rec) * (ROB_NUM * robot._w_mu)

        print(f"- Done! - ")

    for i, robot in enumerate(robots):
        robot.compute_centroid()
    
    # if t in saveTimes:
        # utils.plot_dataset(fig, t, PERIOD, BBOX, FIELD, ax1, ax2, ax3, X1_FIELD, X2_FIELD, X1MESH, X2MESH, robots)
    utils.plot_dataset(fig, t, PERIOD, BBOX, Z, ax1, ax2, ax3, x1_, x2_, _X1, _X2, robots, axes, A)
            
    # Move the robots
    for robot in robots:
        x1, x2 = robot.position + (-K_GAIN*(robot.position - robot.centroid) * D_t)
        robot.move(x1, x2)
    
    # for robot in robots:
    #     robot.save_data()
    
    # for robot in robots:
    #     robot.save_rmse()