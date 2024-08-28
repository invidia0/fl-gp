import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from matplotlib.gridspec import GridSpec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import warnings
import time
import utilities as utils
warnings.filterwarnings("ignore")

class robot:
    def __init__(self, id):
        self.id = id
        self.dataset = np.empty((0, 3))
        self.hyps = np.array([1, 1, 1], dtype=np.float128) # lengthscale, sigma_f, sigma_y
        self.neighbors_hyps = np.empty((0, 3), dtype=np.float128)
        self._p = np.array([0, 0, 0], dtype=np.float128)
        self._mean = np.array([0, 0], dtype=np.float128) # mu (mean)
        self._cov = np.array([0, 0], dtype=np.float128) # sigma**2 (variance)
        self._cov_rec = np.array([0, 0], dtype=np.float128) # sigma**-2 (variance)
        self._w_mu = np.array([0, 0], dtype=np.float128) # w_mu (mean)
        self._w_cov = np.array([0, 0], dtype=np.float128)

    def set_hyps(self, hyps):
        self.hyps = hyps
    
    def set_dataset(self, dataset):
        self.dataset = dataset
    
    def set_neighbors_hyps(self, neighbors_hyps):
        self.neighbors_hyps = neighbors_hyps

    def get_neighbors_hyps(self):
        return self.neighbors_hyps
    
    def get_hyps(self):
        return self.hyps
    
    def get_dataset(self):
        return self.dataset
    
    def get_id(self):
        return self.id

    @property
    def lengthscale(self):
        return self.hyps[0]

    @property
    def sigma_f(self):
        return self.hyps[1]
    
    @property
    def sigma_y(self):
        return self.hyps[2]
    
    @property
    def p(self):
        return self._p
    
    @property
    def mean(self):
        return self._mean
    
    @property
    def cov(self):
        return self._cov
    
    @property
    def cov_rec(self):
        return self._cov_rec

    @p.setter
    def p(self, value):
        self._p = value

    @mean.setter
    def mean(self, value):
        self._mean = value

    @cov.setter
    def cov(self, value):
        self._cov = value
    
    @cov_rec.setter
    def cov_rec(self, value):
        self._cov_rec = value

def plot_ax(ax, data, x1, x2, title, cmap='viridis'):
    """ Plots the data. """
    field_p = ax.contourf(x1, x2, data.reshape(len(x1), len(x2)), cmap=cmap)
    # Divide the plot into 4 regions
    ax.plot([5, 5], [0, 10], 'k-', zorder=1, alpha=0.5)
    ax.plot([0, 10], [5, 5], 'k-', zorder=1, alpha=0.5)
    # Colorbar
    # cbar = plt.colorbar(field_p, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

def plot_drone(ax, x, y, orientation=0, color='k'):
    """ Plots the drone position. """
    # Draw the drone body with the orientation
    p1 = [x + 0.25*cos(orientation), y + 0.25*sin(orientation)]
    p2 = [x - 0.25*sin(orientation), y + 0.25*cos(orientation)]
    p3 = [x - 0.25*cos(orientation), y - 0.25*sin(orientation)]
    p4 = [x + 0.25*sin(orientation), y - 0.25*cos(orientation)]

    # Draw the drone body frame
    ax.plot([p1[0], p3[0]], [p1[1], p3[1]], c=color, zorder=2)
    ax.plot([p2[0], p4[0]], [p2[1], p4[1]], c=color, zorder=2)

    # Draw the drone propellers
    ax.scatter(p1[0], p1[1], c=color, marker='o', s=200, zorder=2)
    ax.scatter(p2[0], p2[1], c=color, marker='o', s=200, zorder=2)
    ax.scatter(p3[0], p3[1], c=color, marker='o', s=200, zorder=2)
    ax.scatter(p4[0], p4[1], c=color, marker='o', s=200, zorder=2)

def RBFKernel(X1: np.ndarray, 
           X2: np.ndarray,
           lengthscale: float=1.0,
           sigma_f: float=1.0) -> np.ndarray:
    """
    Exponentiated Quadratic Kernel
    (https://peterroelants.github.io/posts/gaussian-process-kernels/#Exponentiated-quadratic-kernel)
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)

    RBF = np.exp(- 0.5 * sqdist / lengthscale**2)

    C = sigma_f**2

    return C * RBF

def log_likelihood_grad(robot):
    """ 
    Computes the gradient of the log-likelihood with respect to 
    the hyperparameters: lengthscale, sigma_f, and sigma_y.
    """
    X = robot.get_dataset()[:, :2]
    y = robot.get_dataset()[:, 2]
    y = np.atleast_2d(y).T
    
    # Calculate the RBF kernel
    K = RBFKernel(X, X, sigma_f=robot.sigma_f, lengthscale=robot.lengthscale)
    
    # Add noise variance term
    C_theta = K + robot.sigma_y**2 * np.eye(len(X)) 
    
    C_theta_inv = np.linalg.solve(C_theta, np.eye(len(X)))
    
    # Compute the squared distance matrix
    sqdist = np.sum(X**2, 1, dtype=np.float128).reshape(-1, 1) + np.sum(X**2, 1, dtype=np.float128) - 2 * np.dot(X, X.T)
    
    # Gradients of the covariance matrix with respect to hyperparameters
    dC_lengthscale = K * sqdist / (robot.lengthscale**3)
    dC_theta_f = 2 * K / robot.sigma_f
    dC_theta_y = np.eye(len(X)) * 2 * robot.sigma_y

    # Compute the gradient of the log-likelihood
    common_term = C_theta_inv - (C_theta_inv @ y @ y.T @ C_theta_inv)
    dNLL_lengthscale = 0.5 * np.trace(common_term @ dC_lengthscale, dtype=np.float128)
    dNLL_sigma_f = 0.5 * np.trace(common_term @ dC_theta_f, dtype=np.float128)
    dNLL_sigma_y = 0.5 * np.trace(common_term @ dC_theta_y, dtype=np.float128)

    return np.array([dNLL_lengthscale, dNLL_sigma_f, dNLL_sigma_y], dtype=np.float128)

# Generate data
np.random.seed(0)

area_size = 40
d_field_ = 1
x1_ = np.arange(0, area_size + d_field_, d_field_)
x2_ = np.arange(0, area_size + d_field_, d_field_)
_X1, _X2 = np.meshgrid(x1_, x2_)
mesh = np.vstack([_X1.ravel(), _X2.ravel()]).T

# Generate random means
peaks = 8 # np.random.randint(1, 10)
means = np.random.uniform(low=0, high=area_size, size=(peaks, 2))
sigma = 5
Z = utils.gmm_pdf_array(mesh[:, 0], mesh[:, 1], sigma, means)
Z = Z.reshape(len(x1_), len(x2_))

n_robots = 6
robots = []
for i in range(n_robots):
    robots.append(robot(i))

# Divide the area into vertical regions and assign each robot to a region
regions = []
for i in range(n_robots):
    x1_min = i * (area_size / n_robots)
    x1_max = (i + 1) * (area_size / n_robots)
    x2_min = 0
    x2_max = area_size
    regions.append([x1_min, x1_max, x2_min, x2_max])

# Place the robots in their regions midpoints
robots_poses_ = np.zeros((n_robots, 2))
for i in range(n_robots):
    x1_min, x1_max, x2_min, x2_max = regions[i]
    robots_poses_[i] = [(x1_min + x1_max) / 2, (x2_min + x2_max) / 2]

# For each drone sample some points in its region
n_points = 100
noise = 0.05

for i, robot in enumerate(robots):
    x1_min, x1_max, x2_min, x2_max = regions[i]
    x1 = np.random.uniform(x1_min, x1_max, n_points)
    x2 = np.random.uniform(x2_min, x2_max, n_points)
    X = np.vstack([x1, x2]).T
    y = utils.gmm_pdf_array(x1, x2, sigma, means) + np.random.normal(0, noise, n_points)
    robot.set_dataset(np.hstack([X, np.atleast_2d(y).T]))

# Plot the datasets with different colors
fig, ax_main = plt.subplots()
ax_main.set_aspect('equal')
field_p = ax_main.contour(x1_, x2_, Z, cmap='YlGnBu', zorder=0)
for robot in robots:
    X = robot.get_dataset()[:, :2]
    ax_main.scatter(X[:, 0], X[:, 1], marker='x', alpha=0.5)
plt.grid(alpha=0.2)
# plt.show()

total_dataset = np.empty((0, 3))
for robot in robots:
    total_dataset = np.vstack([total_dataset, robot.get_dataset()])

x_train, y_train = total_dataset[:, :2], total_dataset[:, 2]
y_train = np.atleast_2d(y_train).T
kernel = ConstantKernel() * RBF() + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b")
s_time = time.time()
gp.fit(x_train, y_train)
e_time = time.time()

print(gp.kernel_)

lengthscale_full = gp.kernel_.get_params()['k1__k2__length_scale']
sigma_f_full = np.sqrt(gp.kernel_.get_params()['k1__k1__constant_value'])
sigma_y_full = np.sqrt(gp.kernel_.get_params()['k2__noise_level'])

print(f"Hyperparameters: lengthscale - {lengthscale_full} | sigma_f - {sigma_f_full} | sigma_y - {sigma_y_full}")

# Reset the hyperparameters
for robot in robots:
    robot.set_hyps(np.array([1, 1, 1])) # lengthscale, sigma_f, sigma_y

# Share data with neighbors
# n_share = 10

# dataset_share = np.empty((0, 3))
# for robot in robots:
#     # Get random samples from the dataset
#     idx = np.random.choice(robot.get_dataset().shape[0], n_share, replace=False)
#     dataset_share = np.vstack([dataset_share, robot.get_dataset()[idx]])

# for robot in robots:
#     # Add the shared data to the dataset and avoid duplicates
#     robot.set_dataset(np.vstack([robot.get_dataset(), dataset_share]))
#     robot.set_dataset(np.unique(robot.get_dataset(), axis=0))


""" DEC-gapx-GP (Kontoudis et al.) """
s_end_DEC_gapx = 100
rho = 500
ki = 5000
TOL_ADMM = 1e-3
old_hypers = np.zeros((n_robots, 3)) # Initialize hypers with zeros

s_time = time.time()

# Vector to store the results
results = []
p_s = []
first_terms = []
second_terms = []
third_terms = []
for robot in robots:
    results.append([])
    p_s.append([])
    first_terms.append([])
    second_terms.append([])
    third_terms.append([])

for s in range(s_end_DEC_gapx):
    print(f"*** {s} ***")
    tmp_hyps = np.empty((0, 3), dtype=np.float128) # Temporary store the new hyperparameters
    for robot in robots:
        id = robot.get_id()
        # Take the neighbors' hyperparameters
        neighbors_hyps = np.empty((0, 3), dtype=np.float128)
        for other_robot in robots:
            if other_robot.get_id() != robot.get_id():
                neighbors_hyps = np.vstack([neighbors_hyps, other_robot.get_hyps()], dtype=np.float128)
        n_neighbors = neighbors_hyps.shape[0]

        # Duals (30a) (Consensus)
        sum = np.array([0, 0, 0], dtype=np.float128)
        for k in range(n_neighbors):
            sum = sum + (robot.get_hyps() - neighbors_hyps[k])
        robot.p = robot.p + rho * sum
        # p_s[id].append(robot.p)

        # Primal (34b) (ADMM)
        first_term = rho * np.sum(neighbors_hyps, axis=0, dtype=np.float128)
        second_term = log_likelihood_grad(robot)
        third_term = (ki + n_neighbors * rho) * robot.get_hyps()
        res = (1 / (ki + 2 * n_neighbors * rho)) * (first_term - second_term + third_term - robot.p)
        tmp_hyps = np.vstack([tmp_hyps, res], dtype=np.float128)
        
        # DEBUG
        # print(res)
        # first_terms[id].append(first_term)
        # second_terms[id].append(second_term)
        # third_terms[id].append(third_term)
        # results[id].append(res)
    old_hypers = np.copy([robot.hyps for robot in robots])
    for i, robot in enumerate(robots):
        robot.set_hyps(tmp_hyps[i])



# Plot the results
# fig1, ax = plt.subplots(3, 1)
# for i in range(n_robots):
#     ax[0].plot(np.array(results[i])[:, 0], label=f"Robot {i}", linestyle='solid')
#     ax[1].plot(np.array(results[i])[:, 1], label=f"Robot {i}", linestyle='dotted')
#     ax[2].plot(np.array(results[i])[:, 2], label=f"Robot {i}", linestyle='dashed')
# ax[0].set_title("Lengthscale")
# ax[1].set_title("Sigma_f")
# ax[2].set_title("Sigma_y")
# ax[0].set_xlabel("Iterations")
# ax[1].set_xlabel("Iterations")
# ax[2].set_xlabel("Iterations")
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[0].grid()
# ax[1].grid()
# ax[2].grid()
# fig1.suptitle("RESULTS")
# plt.tight_layout()

# # Plot the duals
# fig2, ax = plt.subplots(3, 1)
# for i in range(n_robots):
#     ax[0].plot(np.array(p_s[i])[:, 0], label=f"Robot {i}", linestyle='solid')
#     ax[1].plot(np.array(p_s[i])[:, 1], label=f"Robot {i}", linestyle='dotted')
#     ax[2].plot(np.array(p_s[i])[:, 2], label=f"Robot {i}", linestyle='dashed')
# ax[0].set_title("Lengthscale")
# ax[1].set_title("Sigma_f")
# ax[2].set_title("Sigma_y")
# ax[0].set_xlabel("Iterations")
# ax[1].set_xlabel("Iterations")
# ax[2].set_xlabel("Iterations")
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[0].grid()
# ax[1].grid()
# ax[2].grid()
# fig2.suptitle("DUALS")

# # Plot the primal terms
# # First term
# fig3, ax = plt.subplots(3, 1)
# for i in range(n_robots):
#     ax[0].plot(np.array(first_terms[i])[:, 0], label=f"Robot {i}", linestyle='solid')
#     ax[1].plot(np.array(first_terms[i])[:, 1], label=f"Robot {i}", linestyle='dotted')
#     ax[2].plot(np.array(first_terms[i])[:, 2], label=f"Robot {i}", linestyle='dashed')
# ax[0].set_title("Lengthscale")
# ax[1].set_title("Sigma_f")
# ax[2].set_title("Sigma_y")
# ax[0].set_xlabel("Iterations")
# ax[1].set_xlabel("Iterations")
# ax[2].set_xlabel("Iterations")
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[0].grid()
# ax[1].grid()
# ax[2].grid()
# fig3.suptitle("First Terms")

# # Second term
# fig4, ax = plt.subplots(3, 1)
# for i in range(n_robots):
#     ax[0].plot(np.array(second_terms[i])[:, 0], label=f"Robot {i}", linestyle='solid')
#     ax[1].plot(np.array(second_terms[i])[:, 1], label=f"Robot {i}", linestyle='dotted')
#     ax[2].plot(np.array(second_terms[i])[:, 2], label=f"Robot {i}", linestyle='dashed')
# ax[0].set_title("Lengthscale")
# ax[1].set_title("Sigma_f")
# ax[2].set_title("Sigma_y")
# ax[0].set_xlabel("Iterations")
# ax[1].set_xlabel("Iterations")
# ax[2].set_xlabel("Iterations")
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[0].grid()
# ax[1].grid()
# ax[2].grid()
# fig4.suptitle("Second Terms")

# # Third term
# fig5, ax = plt.subplots(3, 1)
# for i in range(n_robots):
#     ax[0].plot(np.array(third_terms[i])[:, 0], label=f"Robot {i}", linestyle='solid')
#     ax[1].plot(np.array(third_terms[i])[:, 1], label=f"Robot {i}", linestyle='dotted')
#     ax[2].plot(np.array(third_terms[i])[:, 2], label=f"Robot {i}", linestyle='dashed')
# ax[0].set_title("Lengthscale")
# ax[1].set_title("Sigma_f")
# ax[2].set_title("Sigma_y")
# ax[0].set_xlabel("Iterations")
# ax[1].set_xlabel("Iterations")
# ax[2].set_xlabel("Iterations")
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[0].grid()
# ax[1].grid()
# ax[2].grid()
# fig5.suptitle("Third Terms")

print("Hyperparameters after DEC-gapx-GP:")
for robot in robots:
    print(f"Robot {robot.get_id()} - {robot.get_hyps()}")

# plt.tight_layout()
# plt.show()

# Plot the new predictions for each robot
fig, axes = plt.subplots(2, 3)
for i in range(n_robots):
    ax = axes[i // 3, i % 3]
    ax.set_aspect('equal')
    x_train, y_train = robots[i].get_dataset()[:, :2], robots[i].get_dataset()[:, 2]
    y_train = np.atleast_2d(y_train).T
    mu, std = utils.posterior(mesh, x_train, y_train, lengthscale=robots[i].lengthscale, sigma_f=robots[i].sigma_f, sigma_y=robots[i].sigma_y)
    ax.contour(x1_, x2_, mu.reshape(len(x1_), len(x2_)), cmap='YlGnBu')

fig.suptitle("No DEC-PoE")

""" Implement the filtering algorithm on the dataset """

""" DEC-PoE-GP (Kontoudis et al.) """
delta = n_robots # The maximum degree represents represents the maximum number of neighbors in the graph (fully connected in this case)
eps = 1 / delta
beta = 1 / n_robots
s_end_DAC = 1000

# Compute the local predictions
for robot in robots:
    x_train, y_train = robot.get_dataset()[:, :2], robot.get_dataset()[:, 2]
    y_train = np.atleast_2d(y_train).T
    mu, cov = utils.posterior(mesh, x_train, y_train, lengthscale=robot.lengthscale, sigma_f=robot.sigma_f, sigma_y=robot.sigma_y)
    robot.mean = np.reshape(mu, (len(x1_), len(x2_))) # 41 x 41
    robot.cov =  np.reshape(np.diag(cov), (len(x1_), len(x2_))) # 41 x 41
    robot.cov_rec = np.reshape(1/np.diag(cov), (len(x1_), len(x2_))) # 41 x 41

# Initialize weights
for robot in robots:
    robot._w_mu = beta * robot.cov_rec * robot.mean
    robot._w_cov = beta * robot.cov_rec

for s in range(s_end_DAC):
    print(f"*** {s} ***")
    tmp_w_mu = []
    tmp_w_cov = []
    for robot in robots:
        id = robot.get_id()
        neighbors_w_mu = []
        neighbors_w_cov = []
        for other_robot in robots:
            if other_robot.get_id() != robot.get_id():
                neighbors_w_mu.append(other_robot._w_mu)
                neighbors_w_cov.append(other_robot._w_cov)
        n_neighbors = len(neighbors_w_mu)

        # DAC 1 (Mean)
        sum = np.zeros([len(x1_), len(x2_)], dtype=np.float128)
        for k in range(n_neighbors):
            sum = sum + (neighbors_w_mu[k] - robot._w_mu)
        tmp_w_mu.append(robot._w_mu + eps * sum)

        # DAC 2 (coviance)
        sum = np.zeros([len(x1_), len(x2_)], dtype=np.float128)
        for k in range(n_neighbors):
            sum = sum + (neighbors_w_cov[k] - robot._w_cov)
        tmp_w_cov.append(robot._w_cov + eps * sum)

    for i, robot in enumerate(robots):
        robot._w_mu = tmp_w_mu[i]
        robot._w_cov = tmp_w_cov[i]

for robot in robots:
    robot.cov_rec = n_robots * robot._w_cov
    robot.mean = (1 / robot.cov_rec) * (n_robots * robot._w_mu)

# RMSE with original GP
rmse = np.sqrt(np.mean((Z - robot.mean)**2))
print(f"RMSE: {rmse}")

# Plot the results
fig, axes = plt.subplots(2, 3)
for i in range(n_robots):
    ax = axes[i // 3, i % 3]
    ax.set_aspect('equal')
    ax.contourf(x1_, x2_, robots[i].mean, cmap='YlGnBu')
fig.suptitle("DEC-PoE")


plt.tight_layout()
plt.show()