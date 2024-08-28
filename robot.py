import os
import csv
import warnings
# import gpflow
import numpy as np
import utilities as utils

from scipy.interpolate import RectBivariateSpline
from scipy.spatial import Voronoi
from matplotlib.path import Path
from shapely.geometry import Polygon, Point
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Robot:
    def __init__(self,
                 total_robots: np.uint8,
                 id: np.uint8,
                 x1_init: np.float64, 
                 x2_init: np.float64, 
                 sensing_range: np.float64, 
                 sensor_noise: np.float64, 
                 bbox: np.ndarray,
                 mesh: np.ndarray, 
                 field_delta: np.int8 = 1) -> None:
        
        self._M = total_robots
        self._id = id
        self._position = np.array([x1_init, x2_init], dtype=np.float64)
        self._sensor_noise = sensor_noise
        self._range = sensing_range
        self._observations = np.empty((0, 3), dtype=np.float64)
        self._eval_dataset = np.empty((0, 3), dtype=np.float64)
        self._W_t = 0.0
        self._field = None
        self._field_delta = field_delta
        self._xVals = np.arange(bbox[0], bbox[2] + field_delta, field_delta)
        self._yVals = np.arange(bbox[1], bbox[3] + field_delta, field_delta)
        self._neighbors = np.empty((0, 0))
        self._bbox = bbox
        self._mesh = mesh
        self.__diagram = np.array([[]], dtype=np.float64)
        self._lengthscale = 1.0
        self._sigma_f = 1.0
        self._sigma_y = 1.0
        self._centroid = np.array([0, 0], dtype=np.float64)
        self._nll = 0.0
        self._group = None
        self._p = 0.0
        self._cov = np.empty((0, 0), dtype=np.float64)
        self._cov_rec = np.empty((0, 0), dtype=np.float64)
        self._std = np.empty((0, 0), dtype=np.float64)

        self._w_mu = np.array([], dtype=np.float64)
        self._w_cov = np.array([], dtype=np.float64)
        self._tmp_w_mu = np.array([], dtype=np.float64)
        self._tmp_w_cov = np.array([], dtype=np.float64)
        
        self._x_train = np.empty((0, 2), dtype=np.float64)
        self._y_train = np.empty((0, 1), dtype=np.float64)

        self._dataset = np.empty((0, 3), dtype=np.float64)
        
        self._maxTakeErr = 0.05 / 1.96
        # self._maxRemoveErr = 0.05 / 1.96

    """ Properties """    
    @property
    def id(self) -> np.uint8:
        return self._id

    @property
    def position(self) -> np.ndarray:
        return self._position

    @property
    def range(self) -> np.float64:
        return self._range

    @property
    def sensor_noise(self) -> np.float64:
        return self._sensor_noise

    @property
    def neighbors(self) -> np.ndarray:
        return self._neighbors

    @property
    def observations(self) -> np.ndarray:
        return self._observations

    @property
    def diagram(self) -> np.ndarray:
        return self.__diagram
    
    @property
    def lengthscale(self) -> np.float64:
        return self._lengthscale
      
    @property
    def sigma_f(self) -> np.float64:
        return self._sigma_f

    @property
    def sigma_y(self) -> np.float64:
        return self._sigma_y
    
    @property
    def stored_hyps(self) -> np.ndarray:
        return self._stored_hyps
    
    @property
    def hyps(self) -> np.ndarray:
        return np.array([self._lengthscale, self._sigma_f, self._sigma_y], dtype=np.float64)

    @property
    def centroid(self) -> np.ndarray:
        return self._centroid

    @property
    def mean(self) -> np.ndarray:
        return self._mean
    
    @property
    def cov(self) -> np.ndarray:
        return self._cov

    @property
    def nll(self) -> np.float64:
        return self._nll
    
    @property
    def group(self) -> np.uint8:
        return self._group
    
    @property
    def p(self) -> np.float64:
        return self._p
    
    @property
    def cov_rec(self) -> np.ndarray:
        return self._cov_rec
    
    @property
    def std(self) -> np.ndarray:
        return self._std
    
    @property
    def w_mu(self) -> np.ndarray:
        return self._w_mu
    
    @property
    def w_cov(self) -> np.ndarray:
        return self._w_cov
    
    @property
    def dataset(self) -> np.ndarray:
        return self._dataset
    
    @property
    def tmp_w_mu(self) -> np.ndarray:
        return self._tmp_w_mu
    
    @property
    def tmp_w_cov(self) -> np.ndarray:
        return self._tmp_w_cov

    """ Setters """
    @neighbors.setter
    def neighbors(self, value: np.ndarray) -> None:
        self._neighbors = value
    
    @hyps.setter
    def hyps(self, value: np.ndarray) -> None:
        self._lengthscale = value[0]
        self._sigma_f = value[1]
        self._sigma_y = value[2]

    @group.setter
    def group(self, value: np.uint8) -> None:
        self._group = value

    @p.setter
    def p(self, value: np.float64) -> None:
        self._p = value

    @w_mu.setter
    def w_mu(self, value: np.ndarray) -> None:
        self._w_mu = value

    @w_cov.setter
    def w_cov(self, value: np.ndarray) -> None:
        self._w_cov = value

    @cov.setter
    def cov(self, value: np.ndarray) -> None:
        self._cov = value
    
    @cov_rec.setter
    def cov_rec(self, value: np.ndarray) -> None:
        self._cov_rec = value
    
    @std.setter
    def std(self, value: np.ndarray) -> None:
        self._std = value

    @mean.setter
    def mean(self, value: np.ndarray) -> None:
        self._mean = value

    @tmp_w_mu.setter
    def tmp_w_mu(self, value: np.ndarray) -> None:
        self._tmp_w_mu = value

    @tmp_w_cov.setter
    def tmp_w_cov(self, value: np.ndarray) -> None:
        self._tmp_w_cov = value

    """ Methods """
    def get_dataset(self) -> np.ndarray:
        return self._observations

    def set_Wt(self, value: np.float64) -> None:
        self._W_t = value

    def set_field(self, field: RectBivariateSpline) -> None:
        self._field = field

    def move(self, x1: np.float64, x2: np.float64) -> None:
        self._position = np.array([x1, x2], dtype=np.float64)
         
    def add_sample(self, points, y, time=0.0) -> None:
        """
        Add a sample to the observations.
        """
        values = np.column_stack((points[:, 0], points[:, 1], y))
        values = np.unique(values, axis=0)

        if self._observations.size != 0:
            unique_idx = np.any(np.all(np.abs(self._observations - values[:, None]) > 0.01, axis=1), axis=1)
            self._observations = np.vstack((self._observations, values[unique_idx]))     
        else:
            self._observations = values.copy()

    def update_dataset(self) -> None:
        """
        Update the dataset.
        """
        self.add_sample(self._eval_dataset[:, :2], self._eval_dataset[:, 2])
        self._eval_dataset = np.empty((0, 3), dtype=np.float64)

    def compute_centroid(self) -> None:       
        """ 
        Computes the centroid of the Voronoi region weighted by the spatial process. 
        """
        vertices = self.__diagram
        field_delta = 1
        dA = field_delta**2

        p = Path(vertices)
        bool_val = p.contains_points(self._mesh)
        
        """ Explore-Exploit trade-off """
        mu = self._mean[bool_val.reshape(self._xVals.shape[0], self._yVals.shape[0])]
        std = self._std[bool_val.reshape(self._xVals.shape[0], self._yVals.shape[0])]
        
        weight = std + self._W_t * mu
        weight = np.exp(weight)-1
        
        A = np.sum(weight) * dA
        if A == 0:
            Cx = self._centroid[0]
            Cy = self._centroid[1]
        else:
            Cx = np.sum(self._mesh[:, 0][bool_val] * weight) * dA / A
            Cy = np.sum(self._mesh[:, 1][bool_val] * weight) * dA / A
    
        self._centroid = np.array([Cx, Cy], dtype=np.float64)

    def __voronoi(self, fast=False) -> None:
        """
        Decentralized Bounded Voronoi Computation
        """
        robot_positions = np.array([self._position] + [neighbor.position for neighbor in self._neighbors if np.linalg.norm(neighbor.position - self._position) <= self._range and not np.all(neighbor.position == self._position)])
        
        points_left = np.copy(robot_positions)
        points_left[:, 0] = 2 * self._bbox[0] - points_left[:, 0]
        points_right = np.copy(robot_positions)
        points_right[:, 0] = 2 * self._bbox[2] - points_right[:, 0]
        points_down = np.copy(robot_positions)
        points_down[:, 1] = 2 * self._bbox[1] - points_down[:, 1]
        points_up = np.copy(robot_positions)
        points_up[:, 1] = 2 * self._bbox[3] - points_up[:, 1]
        points = np.vstack((robot_positions, points_left, points_right, points_down, points_up))
        
        # Voronoi diagram
        vor = Voronoi(points)
        vor.filtered_points = robot_positions
        vor.filtered_regions = np.array(vor.regions, dtype=object)[vor.point_region[:vor.npoints//5]]
        vertices = vor.vertices[vor.filtered_regions[0] + [vor.filtered_regions[0][0]], :] # First Site
        
        intersection_points = np.empty((0, 2))
        # Intersect the Voronoi region with the closed polygon centered at the robot position
        polygon = Polygon(vertices)
        if not fast:
            circle = Point(self._position).buffer(self._range * 0.5)
            intersection = polygon.intersection(circle)
            if intersection.is_empty:
                intersection_points = np.empty((0, 2))
            elif intersection.geom_type == 'Polygon':
                intersection_points = np.array(list(intersection.exterior.coords))
            elif intersection.geom_type == 'MultiPolygon':
                intersection_points = np.empty((0, 2))
                for poly in intersection:
                    intersection_points = np.append(intersection_points, list(poly.exterior.coords), axis=0)
        else:
            square = Polygon([(self._position[0] - self._range * 0.5, self._position[1] - self._range * 0.5),
                              (self._position[0] + self._range * 0.5, self._position[1] - self._range * 0.5),
                              (self._position[0] + self._range * 0.5, self._position[1] + self._range * 0.5),
                              (self._position[0] - self._range * 0.5, self._position[1] + self._range * 0.5)])
            intersection = polygon.intersection(square)
            if intersection.is_empty:
                intersection_points = np.empty((0, 2))
            elif intersection.geom_type == 'Polygon':
                intersection_points = np.array(list(intersection.exterior.coords))
            elif intersection.geom_type == 'MultiPolygon':
                intersection_points = np.empty((0, 2))
                for poly in intersection:
                    intersection_points = np.append(intersection_points, list(poly.exterior.coords), axis=0)

        # Add the intersection points to the diagram list
        self.__diagram = intersection_points
    
    def compute_voronoi(self) -> np.ndarray:
        """
        Computes the centroid of the observations.
        """
        self.__voronoi()

    def predict(self, values: np.ndarray) -> tuple:
        """
        Compute the posterior prediction with the robot parameters.
        """
        self._x_train = self._observations[:, :2]
        self._y_train = self._observations[:, 2].reshape(-1, 1)
        
        mu, cov = utils.posterior(values,
                            self._x_train,
                            self._y_train,
                            lengthscale=self._lengthscale,
                            sigma_f=self._sigma_f,
                            sigma_y=self._sigma_y)
        return mu, cov
    
    def filter_dataset(self) -> None:
        """ Nyström approximation """
        self._x_train = self._observations[:, :2]
        self._y_train = self._observations[:, 2].reshape(-1, 1)

        K = utils.RBFKernel(self._x_train, self._x_train, self._lengthscale, self._sigma_f) + self._sigma_y**2 * np.eye(self._x_train.shape[0])
        eigvals, eigvecs = np.linalg.eigh(K)
        sorted_indices = np.argsort(eigvals)[::-1] # Sort the eigenvalues in descending order
        eigvals = eigvals[sorted_indices] # Sort the eigenvalues
        eigvecs = eigvecs[:, sorted_indices] # Sort the eigenvectors
        n_top = np.argmax(np.cumsum(eigvals) / np.sum(eigvals) >= 0.95) + 1 # Find the number of components needed to explain 95% of the variance
        top_eigvecs = eigvecs[:, :n_top]
        # Initialize a set to store all influential points
        all_influential_points = set()

        # For each of these eigenvectors, find the data points corresponding to the largest magnitude entries.
        # These points are the "influential points" for the eigenvector.
        for i in range(n_top):
            # Compute absolute values of the eigenvector
            abs_eigvec = np.abs(top_eigvecs[:, i])
            
            # Set a threshold (e.g., 50% of the maximum value)
            threshold = 0.95  * np.max(abs_eigvec)
            
            # Find points above the threshold
            influential_points = np.where(abs_eigvec > threshold)[0]

            all_influential_points.update(influential_points)

        # Highlight the influential points in the covariance matrix
        influential_points_list = list(all_influential_points)

        # Filter the dataset and store the influential points and their corresponding values into self._dataset
        self._dataset = self._observations[influential_points_list]

    def sense(self, points, value, first=False):
        """
        Sense the environment and add the sample to the observations if the sample contributes to the knowledge.
        """
        points = np.atleast_2d(points)
        value = np.atleast_1d(value)
        
        if not first:
            mu, cov = self.predict(points)

            std = np.sqrt(np.diag(cov))

            mask = (std > self.sigma_y + self._maxTakeErr * self._mu_max).reshape(-1)
            
            points = points[mask, :]
            value = value[mask]
            
            self._eval_dataset = np.vstack((self._eval_dataset, np.column_stack((points, value))))
        else:
            self._eval_dataset = np.vstack((self._eval_dataset, np.column_stack((points, value))))
        # self._eval_dataset = np.vstack((self._eval_dataset, np.column_stack((points, value))))
            
    def update_estimate(self):
        mu, cov = self.predict(self._mesh)
        self._mean = np.reshape(mu, (self._xVals.shape[0], self._yVals.shape[0]))
        self._cov = np.reshape(np.diag(cov), (self._xVals.shape[0], self._yVals.shape[0]))
        self._cov_rec = np.reshape(1/np.diag(cov), (self._xVals.shape[0], self._yVals.shape[0]))
        # self._std = np.reshape(np.sqrt(np.diag(cov)), (self._xVals.shape[0], self._yVals.shape[0]))
        self._mu_max = np.max(self._mean)
        self._mu_min = np.min(self._mean)
        self._delta = self._mu_max - self._mu_min

    def clear_observations(self) -> None:
        values = np.array([self._observations[:, 0], self._observations[:, 1]]).T
        mu, std = self.predict(values)
        std = std.reshape(-1, 1)

        mask = (std > self._maxRemoveErr * self._mu_max).reshape(-1)
        
        self._observations = self._observations[~mask, :]
        # self._time_samples_array = self._time_samples_array[~mask]
        print(f"Robot {self._id} removed {np.sum(mask)} observations")

    def save_data(self) -> None:
        """
        Save the data for each robot.
        """
        # Save the current position and centroid in a csv file
        header = ["x", "y", "centroid_x", "centroid_y"]
        with open(f"{self._path}/robot-{self._id}.csv", "a", newline="") as file:
            writer = csv.writer(file)
            if os.stat(f"{self._path}/robot-{self._id}.csv").st_size == 0:
                writer.writerow(header)
            writer.writerow([self._position[0], self._position[1], self._centroid[0], self._centroid[1]])
        
        if self._sparse:
            # Save the mu_m, A_m, K_mm_inv in a csv file
            header = ["mu_m", "A_m", "K_mm_inv"]
            with open(f"{self._path}/sparse-{self._id}.csv", "a", newline="") as file:
                writer = csv.writer(file)
                if os.stat(f"{self._path}/sparse-{self._id}.csv").st_size == 0:
                    writer.writerow(header)
                writer.writerow([self._t_actual])
                writer.writerow(self._mu_m)
                writer.writerow(self._A_m)
                writer.writerow(self._K_mm_inv)

        header = ["lengthscale", "sigma_f", "sigma_t", "sigma_y"]
        with open(f"{self._path}/hyps-{self._id}.csv", "a", newline="") as file:
            writer = csv.writer(file)
            if os.stat(f"{self._path}/hyps-{self._id}.csv").st_size == 0:
                writer.writerow(header)
            writer.writerow(self.hyps)
        
        # Save the observations in a csv file
        with open(f"{self._path}/observations-{self._id}.csv", "a", newline="") as file:
            writer = csv.writer(file)
            # Write the actual time
            writer.writerow([self._t_actual])
            writer.writerow(self._observations[:, 0])
            writer.writerow(self._observations[:, 1])
            writer.writerow(self._observations[:, 2])
            writer.writerow(self._observations[:, 3])
            
        # Save the computation time in a csv file
        with open(f"{self._path}/computation-time-{self._id}.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([self._compTime])
        
    def save_rmse(self) -> None:
        rmse = np.sqrt(np.mean((self._mean - self._field.T)**2))
        with open(f"{self._path}/rmse-{self._id}.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([rmse])