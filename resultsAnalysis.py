import numpy as np
import matplotlib.pyplot as plt
import utilities as utils
from pathlib import Path
from scipy.interpolate import griddata
import xarray as xr
import seaborn as sns

# np.random.seed(0)

# area_size = 40
# x_inf, y_inf = 0, 0
# x_sup, y_sup = area_size, area_size
# BBOX = [x_inf, y_inf, x_sup, y_sup]
# d_field_ = 1
# x1_ = np.arange(0, area_size + d_field_, d_field_)
# x2_ = np.arange(0, area_size + d_field_, d_field_)
# _X1, _X2 = np.meshgrid(x1_, x2_)
# mesh = np.vstack([_X1.ravel(), _X2.ravel()]).T

# robNum = 6

# sensRange = np.sqrt(((x_sup * y_sup) * 1.0 / robNum) / np.pi) * 2

# # Generate random means
# peaks = 2 # np.random.randint(1, 10)
# means = np.random.uniform(low=0, high=area_size, size=(peaks, 2))
# sigma = 6
# Z = utils.gmm_pdf_array(mesh[:, 0], mesh[:, 1], sigma, means, flag_normalize=False)
# Z = Z.reshape(len(x1_), len(x2_))


""" Dataset """
# Generate data
np.random.seed(0)

area_size = 40
x_inf, y_inf = 0, 0
x_sup, y_sup = area_size, area_size
BBOX = [x_inf, y_inf, x_sup, y_sup]
d_field_ = 1
x1_ = np.arange(0, area_size, d_field_)
x2_ = np.arange(0, area_size, d_field_)
_X1, _X2 = np.meshgrid(x1_, x2_)
mesh = np.vstack([_X1.ravel(), _X2.ravel()]).T

# Load the dataset and select the region of interest
dataset_path = Path('dataset.nc')
ds = xr.open_dataset(dataset_path)
sst_region = ds['analysed_sst'].sel(lat=slice(22, 23), lon=slice(-115, -114)).isel(time=0).values

# Convert from Kelvin to Celsius
sst_region = (sst_region - 273.15).astype(np.float32)

# Define original and target grids
original_shape = sst_region.shape
target_size = (area_size, area_size)
original_grid = np.meshgrid(np.arange(original_shape[1]), np.arange(original_shape[0]))
target_grid = np.meshgrid(np.linspace(0, original_shape[1]-1, target_size[1]),
                          np.linspace(0, original_shape[0]-1, target_size[0]))

# Interpolate SST data to the target grid
sst_interpolated = griddata(
    np.column_stack([g.flatten() for g in original_grid]),
    sst_region.flatten(),
    np.column_stack([g.flatten() for g in target_grid]),
    method='linear'
).reshape(target_size)

# Min-Max Normalization
field = (sst_interpolated - sst_interpolated.min()) / (sst_interpolated.max() - sst_interpolated.min())

general = 0
metrics = 1
time_comparison = 0

fontsize = 22
fontdict = {'weight': 'bold', 
            'size': fontsize, 
            'color': 'black',
            'family': 'serif'}
alpha = 0.8
cmap = 'hot'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# 6 ROBOTS
path_gen = Path('6robs')
sims = ['sim-1', 'sim-2', 'sim-3', 'sim-4', 'sim-5', 'sim-6']
# sims = ['sim-4', 'sim-5', 'sim-6']

rmseHistory_6 = []
nlpdHistory_6 = []
observedHistory_6 = []
filteredHistory_6 = []
DEC_gapx_time_hist_6 = []
DAC_time_hist_6 = []
timeHistory_6 = []
meanHistory_6 = []
stdHistory_6 = []
robotHistory_6 = []

for sim in sims:
    path = path_gen / sim
    robotHistory = np.load(path / "robotHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    observedHistory = np.load(path / "observedHistory.npy")
    filteredHistory = np.load(path / "filteredHistory.npy")
    DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
    DAC_time_hist = np.load(path / "DAC_time_hist.npy")
    timeHistory = np.load(path / "timeHistory.npy")
    meanHistory = np.load(path / "meanHistory.npy")
    stdHistory = np.load(path / "stdHistory.npy")

    rmseHistory_6.append(rmseHistory)
    robotHistory_6.append(robotHistory)
    nlpdHistory_6.append(nlpdHistory)
    observedHistory_6.append(observedHistory)
    filteredHistory_6.append(filteredHistory)
    DEC_gapx_time_hist_6.append(DEC_gapx_time_hist)
    DAC_time_hist_6.append(DAC_time_hist)
    timeHistory_6.append(timeHistory)
    meanHistory_6.append(meanHistory)
    stdHistory_6.append(stdHistory)

# 12 ROBOTS
path_gen = Path('12robs')
sims = ['sim-1', 'sim-2', 'sim-3']
# sims = ['sim-4', 'sim-5', 'sim-6']

rmseHistory_12 = []
nlpdHistory_12 = []
observedHistory_12 = []
filteredHistory_12 = []
DEC_gapx_time_hist_12 = []
DAC_time_hist_12 = []
timeHistory_12 = []
meanHistory_12 = []
stdHistory_12 = []
robotHistory_12 = []

for sim in sims:
    path = path_gen / sim
    robotHistory = np.load(path / "robotHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    observedHistory = np.load(path / "observedHistory.npy")
    filteredHistory = np.load(path / "filteredHistory.npy")
    DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
    DAC_time_hist = np.load(path / "DAC_time_hist.npy")
    timeHistory = np.load(path / "timeHistory.npy")
    meanHistory = np.load(path / "meanHistory.npy")
    stdHistory = np.load(path / "stdHistory.npy")

    rmseHistory_12.append(rmseHistory)
    robotHistory_12.append(robotHistory)
    nlpdHistory_12.append(nlpdHistory)
    observedHistory_12.append(observedHistory)
    filteredHistory_12.append(filteredHistory)
    DEC_gapx_time_hist_12.append(DEC_gapx_time_hist)
    DAC_time_hist_12.append(DAC_time_hist)
    timeHistory_12.append(timeHistory)
    meanHistory_12.append(meanHistory)
    stdHistory_12.append(stdHistory)

# 18 ROBOTS
path_gen = Path('18robs')
sims = ['sim-1', 'sim-2', 'sim-3']
# sims = ['sim-4', 'sim-5', 'sim-6']

rmseHistory_18 = []
nlpdHistory_18 = []
observedHistory_18 = []
filteredHistory_18 = []
DEC_gapx_time_hist_18 = []
DAC_time_hist_18 = []
timeHistory_18 = []
meanHistory_18 = []
stdHistory_18 = []
robotHistory_18 = []

for sim in sims:
    path = path_gen / sim
    robotHistory = np.load(path / "robotHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    observedHistory = np.load(path / "observedHistory.npy")
    filteredHistory = np.load(path / "filteredHistory.npy")
    DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
    DAC_time_hist = np.load(path / "DAC_time_hist.npy")
    timeHistory = np.load(path / "timeHistory.npy")
    meanHistory = np.load(path / "meanHistory.npy")
    stdHistory = np.load(path / "stdHistory.npy")

    rmseHistory_18.append(rmseHistory)
    robotHistory_18.append(robotHistory)
    nlpdHistory_18.append(nlpdHistory)
    observedHistory_18.append(observedHistory)
    filteredHistory_18.append(filteredHistory)
    DEC_gapx_time_hist_18.append(DEC_gapx_time_hist)
    DAC_time_hist_18.append(DAC_time_hist)
    timeHistory_18.append(timeHistory)
    meanHistory_18.append(meanHistory)
    stdHistory_18.append(stdHistory)

# 24 ROBOTS
path_gen = Path('24robs')
sims = ['sim-1', 'sim-2', 'sim-3']
# sims = ['sim-4', 'sim-5', 'sim-6']

rmseHistory_24 = []
nlpdHistory_24 = []
observedHistory_24 = []
filteredHistory_24 = []
DEC_gapx_time_hist_24 = []
DAC_time_hist_24 = []
timeHistory_24 = []
meanHistory_24 = []
stdHistory_24 = []
robotHistory_24 = []

for sim in sims:
    path = path_gen / sim
    robotHistory = np.load(path / "robotHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    observedHistory = np.load(path / "observedHistory.npy")
    filteredHistory = np.load(path / "filteredHistory.npy")
    DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
    DAC_time_hist = np.load(path / "DAC_time_hist.npy")
    timeHistory = np.load(path / "timeHistory.npy")
    meanHistory = np.load(path / "meanHistory.npy")
    stdHistory = np.load(path / "stdHistory.npy")

    rmseHistory_24.append(rmseHistory)
    robotHistory_24.append(robotHistory)
    nlpdHistory_24.append(nlpdHistory)
    observedHistory_24.append(observedHistory)
    filteredHistory_24.append(filteredHistory)
    DEC_gapx_time_hist_24.append(DEC_gapx_time_hist)
    DAC_time_hist_24.append(DAC_time_hist)
    timeHistory_24.append(timeHistory)
    meanHistory_24.append(meanHistory)
    stdHistory_24.append(stdHistory)

# RAL 6 ROBOTS
path_gen = Path('6robs_ral')
sims = ['sim-1', 'sim-2', 'sim-3']
# sims = ['sim-4', 'sim-5', 'sim-6']

rmseHistory_6_ral = []
nlpdHistory_6_ral = []
observedHistory_6_ral = []
filteredHistory_6_ral = []
DEC_gapx_time_hist_6_ral = []
DAC_time_hist_6_ral = []
timeHistory_6_ral = []
meanHistory_6_ral = []
stdHistory_6_ral = []
robotHistory_6_ral = []

for sim in sims:
    path = path_gen / sim
    robotHistory = np.load(path / "robotHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    observedHistory = np.load(path / "observedHistory.npy")
    filteredHistory = np.load(path / "filteredHistory.npy")
    DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
    DAC_time_hist = np.load(path / "DAC_time_hist.npy")
    timeHistory = np.load(path / "timeHistory.npy")
    meanHistory = np.load(path / "meanHistory.npy")
    stdHistory = np.load(path / "stdHistory.npy")

    rmseHistory_6_ral.append(rmseHistory)
    robotHistory_6_ral.append(robotHistory)
    nlpdHistory_6_ral.append(nlpdHistory)
    observedHistory_6_ral.append(observedHistory)
    filteredHistory_6_ral.append(filteredHistory)
    DEC_gapx_time_hist_6_ral.append(DEC_gapx_time_hist)
    DAC_time_hist_6_ral.append(DAC_time_hist)
    timeHistory_6_ral.append(timeHistory)
    meanHistory_6_ral.append(meanHistory)
    stdHistory_6_ral.append(stdHistory)

# RAL 12 ROBOTS
path_gen = Path('12robs_ral')
sims = ['sim-1', 'sim-2', 'sim-3']
# sims = ['sim-4', 'sim-5', 'sim-6']

rmseHistory_12_ral = []
nlpdHistory_12_ral = []
observedHistory_12_ral = []
filteredHistory_12_ral = []
DEC_gapx_time_hist_12_ral = []
DAC_time_hist_12_ral = []
timeHistory_12_ral = []
meanHistory_12_ral = []
stdHistory_12_ral = []
robotHistory_12_ral = []

for sim in sims:
    path = path_gen / sim
    robotHistory = np.load(path / "robotHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    observedHistory = np.load(path / "observedHistory.npy")
    filteredHistory = np.load(path / "filteredHistory.npy")
    DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
    DAC_time_hist = np.load(path / "DAC_time_hist.npy")
    timeHistory = np.load(path / "timeHistory.npy")
    meanHistory = np.load(path / "meanHistory.npy")
    stdHistory = np.load(path / "stdHistory.npy")

    rmseHistory_12_ral.append(rmseHistory)
    robotHistory_12_ral.append(robotHistory)
    nlpdHistory_12_ral.append(nlpdHistory)
    observedHistory_12_ral.append(observedHistory)
    filteredHistory_12_ral.append(filteredHistory)
    DEC_gapx_time_hist_12_ral.append(DEC_gapx_time_hist)
    DAC_time_hist_12_ral.append(DAC_time_hist)
    timeHistory_12_ral.append(timeHistory)
    meanHistory_12_ral.append(meanHistory)
    stdHistory_12_ral.append(stdHistory)

# RAL 18 ROBOTS
path_gen = Path('18robs_ral')
sims = ['sim-1', 'sim-2', 'sim-3']
# sims = ['sim-4', 'sim-5', 'sim-6']

rmseHistory_18_ral = []
nlpdHistory_18_ral = []
observedHistory_18_ral = []
filteredHistory_18_ral = []
DEC_gapx_time_hist_18_ral = []
DAC_time_hist_18_ral = []
timeHistory_18_ral = []
meanHistory_18_ral = []
stdHistory_18_ral = []
robotHistory_18_ral = []

for sim in sims:
    path = path_gen / sim
    robotHistory = np.load(path / "robotHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    observedHistory = np.load(path / "observedHistory.npy")
    filteredHistory = np.load(path / "filteredHistory.npy")
    DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
    DAC_time_hist = np.load(path / "DAC_time_hist.npy")
    timeHistory = np.load(path / "timeHistory.npy")
    meanHistory = np.load(path / "meanHistory.npy")
    stdHistory = np.load(path / "stdHistory.npy")

    rmseHistory_18_ral.append(rmseHistory)
    robotHistory_18_ral.append(robotHistory)
    nlpdHistory_18_ral.append(nlpdHistory)
    observedHistory_18_ral.append(observedHistory)
    filteredHistory_18_ral.append(filteredHistory)
    DEC_gapx_time_hist_18_ral.append(DEC_gapx_time_hist)
    DAC_time_hist_18_ral.append(DAC_time_hist)
    timeHistory_18_ral.append(timeHistory)
    meanHistory_18_ral.append(meanHistory)
    stdHistory_18_ral.append(stdHistory)

# RAL 24 ROBOTS
path_gen = Path('24robs_ral')
sims = ['sim-1', 'sim-2']
# sims = ['sim-4', 'sim-5', 'sim-6']

rmseHistory_24_ral = []
nlpdHistory_24_ral = []
observedHistory_24_ral = []
filteredHistory_24_ral = []
DEC_gapx_time_hist_24_ral = []
DAC_time_hist_24_ral = []
timeHistory_24_ral = []
meanHistory_24_ral = []
stdHistory_24_ral = []
robotHistory_24_ral = []

for sim in sims:
    path = path_gen / sim
    robotHistory = np.load(path / "robotHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    observedHistory = np.load(path / "observedHistory.npy")
    filteredHistory = np.load(path / "filteredHistory.npy")
    DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
    DAC_time_hist = np.load(path / "DAC_time_hist.npy")
    timeHistory = np.load(path / "timeHistory.npy")
    meanHistory = np.load(path / "meanHistory.npy")
    stdHistory = np.load(path / "stdHistory.npy")

    rmseHistory_24_ral.append(rmseHistory)
    robotHistory_24_ral.append(robotHistory)
    nlpdHistory_24_ral.append(nlpdHistory)
    observedHistory_24_ral.append(observedHistory)
    filteredHistory_24_ral.append(filteredHistory)
    DEC_gapx_time_hist_24_ral.append(DEC_gapx_time_hist)
    DAC_time_hist_24_ral.append(DAC_time_hist)
    timeHistory_24_ral.append(timeHistory)
    meanHistory_24_ral.append(meanHistory)
    stdHistory_24_ral.append(stdHistory)

robNum = robotHistory.shape[0]
sensRange = np.sqrt(((x_sup * y_sup) * 1.0 / robNum) / np.pi) * 2

# # Average RMSE for 6, 12, 18, 24 robots
# rmse_mean_6 = np.mean(np.mean(np.mean(rmseHistory_6, axis=0), axis=0), axis=0)
# rmse_std_6 = np.mean(np.std(rmseHistory_6, axis=0))
# print(f'RMSE 6: {rmse_mean_6:.2f} ± {rmse_std_6:.2f}')

# rmse_mean_12 = np.mean(np.mean(np.mean(rmseHistory_12, axis=0), axis=0), axis=0)
# rmse_std_12 = np.mean(np.std(rmseHistory_12, axis=0))
# print(f'RMSE 12: {rmse_mean_12:.2f} ± {rmse_std_12:.2f}')

# rmse_mean_18 = np.mean(np.mean(np.mean(rmseHistory_18, axis=0), axis=0), axis=0)
# rmse_std_18 = np.mean(np.std(rmseHistory_18, axis=0))
# print(f'RMSE 18: {rmse_mean_18:.2f} ± {rmse_std_18:.2f}')

# rmse_mean_24 = np.mean(np.mean(np.mean(rmseHistory_24, axis=0), axis=0), axis=0)
# rmse_std_24 = np.mean(np.std(rmseHistory_24, axis=0))
# print(f'RMSE 24: {rmse_mean_24:.2f} ± {rmse_std_24:.2f}')

# # Average NLPD for 6, 12, 18, 24 robots
# nlpd_mean_6 = np.mean(np.mean(np.mean(nlpdHistory_6, axis=0), axis=0), axis=0)
# nlpd_std_6 = np.mean(np.std(nlpdHistory_6, axis=0))
# print(f'NLPD 6: {nlpd_mean_6:.2f} ± {nlpd_std_6:.2f}')

# nlpd_mean_12 = np.mean(np.mean(np.mean(nlpdHistory_12, axis=0), axis=0), axis=0)
# nlpd_std_12 = np.mean(np.std(nlpdHistory_12, axis=0))
# print(f'NLPD 12: {nlpd_mean_12:.2f} ± {nlpd_std_12:.2f}')

# nlpd_mean_18 = np.mean(np.mean(np.mean(nlpdHistory_18, axis=0), axis=0), axis=0)
# nlpd_std_18 = np.mean(np.std(nlpdHistory_18, axis=0))
# print(f'NLPD 18: {nlpd_mean_18:.2f} ± {nlpd_std_18:.2f}')

# nlpd_mean_24 = np.mean(np.mean(np.mean(nlpdHistory_24, axis=0), axis=0), axis=0)
# nlpd_std_24 = np.mean(np.std(nlpdHistory_24, axis=0))
# print(f'NLPD 24: {nlpd_mean_24:.2f} ± {nlpd_std_24:.2f}')

# # Average observed and filtered data for 6, 12, 18, 24 robots
# observed_mean_6 = np.mean(np.mean(np.mean(observedHistory_6, axis=0), axis=0), axis=0)
# observed_std_6 = np.mean(np.std(observedHistory_6, axis=0))
# print(f'Observed 6: {observed_mean_6:.2f} ± {observed_std_6:.2f}')

# filtered_mean_6 = np.mean(np.mean(np.mean(filteredHistory_6, axis=0), axis=0), axis=0)
# filtered_std_6 = np.mean(np.std(filteredHistory_6, axis=0))
# print(f'Filtered 6: {filtered_mean_6:.2f} ± {filtered_std_6:.2f}')

# observed_mean_12 = np.mean(np.mean(np.mean(observedHistory_12, axis=0), axis=0), axis=0)
# observed_std_12 = np.mean(np.std(observedHistory_12, axis=0))
# print(f'Observed 12: {observed_mean_12:.2f} ± {observed_std_12:.2f}')

# filtered_mean_12 = np.mean(np.mean(np.mean(filteredHistory_12, axis=0), axis=0), axis=0)
# filtered_std_12 = np.mean(np.std(filteredHistory_12, axis=0))
# print(f'Filtered 12: {filtered_mean_12:.2f} ± {filtered_std_12:.2f}')

# observed_mean_18 = np.mean(np.mean(np.mean(observedHistory_18, axis=0), axis=0), axis=0)
# observed_std_18 = np.mean(np.std(observedHistory_18, axis=0))
# print(f'Observed 18: {observed_mean_18:.2f} ± {observed_std_18:.2f}')

# filtered_mean_18 = np.mean(np.mean(np.mean(filteredHistory_18, axis=0), axis=0), axis=0)
# filtered_std_18 = np.mean(np.std(filteredHistory_18, axis=0))
# print(f'Filtered 18: {filtered_mean_18:.2f} ± {filtered_std_18:.2f}')

# observed_mean_24 = np.mean(np.mean(np.mean(observedHistory_24, axis=0), axis=0), axis=0)
# observed_std_24 = np.mean(np.std(observedHistory_24, axis=0))
# print(f'Observed 24: {observed_mean_24:.2f} ± {observed_std_24:.2f}')

# filtered_mean_24 = np.mean(np.mean(np.mean(filteredHistory_24, axis=0), axis=0), axis=0)
# filtered_std_24 = np.mean(np.std(filteredHistory_24, axis=0))
# print(f'Filtered 24: {filtered_mean_24:.2f} ± {filtered_std_24:.2f}')

# # Average Time for 6, 12, 18, 24 robots
# time_mean_6 = np.mean(np.mean(timeHistory_6, axis=0), axis=0)
# time_std_6 = np.mean(np.std(timeHistory_6, axis=0))
# print(f'Time 6: {time_mean_6[-1]:.2f} ± {time_std_6:.2f}')

# time_mean_12 = np.mean(np.mean(timeHistory_12, axis=0), axis=0)
# time_std_12 = np.mean(np.std(timeHistory_12, axis=0))
# print(f'Time 12: {time_mean_12[-1]:.2f} ± {time_std_12:.2f}')

# time_mean_18 = np.mean(np.mean(timeHistory_18, axis=0), axis=0)
# time_std_18 = np.mean(np.std(timeHistory_18, axis=0))
# print(f'Time 18: {time_mean_18[-1]:.2f} ± {time_std_18:.2f}')

# time_mean_24 = np.mean(np.mean(timeHistory_24, axis=0), axis=0)
# time_std_24 = np.mean(np.std(timeHistory_24, axis=0))
# print(f'Time 24: {time_mean_24[-1]:.2f} ± {time_std_24:.2f}')


# # RAL Data
# # Average RMSE for 6, 12, 18, 24 robots
# rmse_mean_6_ral = np.mean(np.mean(np.mean(rmseHistory_6_ral, axis=0), axis=0), axis=0)
# rmse_std_6_ral = np.mean(np.std(rmseHistory_6_ral, axis=0))
# print(f'RMSE 6 RAL: {rmse_mean_6_ral:.2f} ± {rmse_std_6_ral:.2f}')

# rmse_mean_12_ral = np.mean(np.mean(np.mean(rmseHistory_12_ral, axis=0), axis=0), axis=0)
# rmse_std_12_ral = np.mean(np.std(rmseHistory_12_ral, axis=0))
# print(f'RMSE 12 RAL: {rmse_mean_12_ral:.2f} ± {rmse_std_12_ral:.2f}')

# rmse_mean_18_ral = np.mean(np.mean(np.mean(rmseHistory_18_ral, axis=0), axis=0), axis=0)
# rmse_std_18_ral = np.mean(np.std(rmseHistory_18_ral, axis=0))
# print(f'RMSE 18 RAL: {rmse_mean_18_ral:.2f} ± {rmse_std_18_ral:.2f}')

# rmse_mean_24_ral = np.mean(np.mean(np.mean(rmseHistory_24_ral, axis=0), axis=0), axis=0)
# rmse_std_24_ral = np.mean(np.std(rmseHistory_24_ral, axis=0))
# print(f'RMSE 24 RAL: {rmse_mean_24_ral:.2f} ± {rmse_std_24_ral:.2f}')

# # Average NLPD for 6, 12, 18, 24 robots
# nlpd_mean_6_ral = np.mean(np.mean(np.mean(nlpdHistory_6_ral, axis=0), axis=0), axis=0)
# nlpd_std_6_ral = np.mean(np.std(nlpdHistory_6_ral, axis=0))
# print(f'NLPD 6 RAL: {nlpd_mean_6_ral:.2f} ± {nlpd_std_6_ral:.2f}')

# nlpd_mean_12_ral = np.mean(np.mean(np.mean(nlpdHistory_12_ral, axis=0), axis=0), axis=0)
# nlpd_std_12_ral = np.mean(np.std(nlpdHistory_12_ral, axis=0))
# print(f'NLPD 12 RAL: {nlpd_mean_12_ral:.2f} ± {nlpd_std_12_ral:.2f}')

# nlpd_mean_18_ral = np.mean(np.mean(np.mean(nlpdHistory_18_ral, axis=0), axis=0), axis=0)
# nlpd_std_18_ral = np.mean(np.std(nlpdHistory_18_ral, axis=0))
# print(f'NLPD 18 RAL: {nlpd_mean_18_ral:.2f} ± {nlpd_std_18_ral:.2f}')

# nlpd_mean_24_ral = np.mean(np.mean(np.mean(nlpdHistory_24_ral, axis=0), axis=0), axis=0)
# nlpd_std_24_ral = np.mean(np.std(nlpdHistory_24_ral, axis=0))
# print(f'NLPD 24 RAL: {nlpd_mean_24_ral:.2f} ± {nlpd_std_24_ral:.2f}')

# # Average observed and filtered data for 6, 12, 18, 24 robots
# observed_mean_6_ral = np.mean(np.mean(np.mean(observedHistory_6_ral, axis=0), axis=0), axis=0)
# observed_std_6_ral = np.mean(np.std(observedHistory_6_ral, axis=0))
# print(f'Observed 6 RAL: {observed_mean_6_ral:.2f} ± {observed_std_6_ral:.2f}')

# filtered_mean_6_ral = np.mean(np.mean(np.mean(filteredHistory_6_ral, axis=0), axis=0), axis=0)
# filtered_std_6_ral = np.mean(np.std(filteredHistory_6_ral, axis=0))
# print(f'Filtered 6 RAL: {filtered_mean_6_ral:.2f} ± {filtered_std_6_ral:.2f}')

# observed_mean_12_ral = np.mean(np.mean(np.mean(observedHistory_12_ral, axis=0), axis=0), axis=0)
# observed_std_12_ral = np.mean(np.std(observedHistory_12_ral, axis=0))
# print(f'Observed 12 RAL: {observed_mean_12_ral:.2f} ± {observed_std_12_ral:.2f}')

# filtered_mean_12_ral = np.mean(np.mean(np.mean(filteredHistory_12_ral, axis=0), axis=0), axis=0)
# filtered_std_12_ral = np.mean(np.std(filteredHistory_12_ral, axis=0))
# print(f'Filtered 12 RAL: {filtered_mean_12_ral:.2f} ± {filtered_std_12_ral:.2f}')

# observed_mean_18_ral = np.mean(np.mean(np.mean(observedHistory_18_ral, axis=0), axis=0), axis=0)
# observed_std_18_ral = np.mean(np.std(observedHistory_18_ral, axis=0))
# print(f'Observed 18 RAL: {observed_mean_18_ral:.2f} ± {observed_std_18_ral:.2f}')

# filtered_mean_18_ral = np.mean(np.mean(np.mean(filteredHistory_18_ral, axis=0), axis=0), axis=0)
# filtered_std_18_ral = np.mean(np.std(filteredHistory_18_ral, axis=0))
# print(f'Filtered 18 RAL: {filtered_mean_18_ral:.2f} ± {filtered_std_18_ral:.2f}')

# observed_mean_24_ral = np.mean(np.mean(np.mean(observedHistory_24_ral, axis=0), axis=0), axis=0)
# observed_std_24_ral = np.mean(np.std(observedHistory_24_ral, axis=0))
# print(f'Observed 24 RAL: {observed_mean_24_ral:.2f} ± {observed_std_24_ral:.2f}')

# filtered_mean_24_ral = np.mean(np.mean(np.mean(filteredHistory_24_ral, axis=0), axis=0), axis=0)
# filtered_std_24_ral = np.mean(np.std(filteredHistory_24_ral, axis=0))
# print(f'Filtered 24 RAL: {filtered_mean_24_ral:.2f} ± {filtered_std_24_ral:.2f}')

# # Average Time for 6, 12, 18, 24 robots
# time_mean_6_ral = np.mean(np.mean(timeHistory_6_ral, axis=0), axis=0)
# time_std_6_ral = np.mean(np.std(timeHistory_6_ral, axis=0))
# print(f'Time 6 RAL: {time_mean_6_ral[-1]:.2f} ± {time_std_6_ral:.2f}')

# time_mean_12_ral = np.mean(np.mean(timeHistory_12_ral, axis=0), axis=0)
# time_std_12_ral = np.mean(np.std(timeHistory_12_ral, axis=0))
# print(f'Time 12 RAL: {time_mean_12_ral[-1]:.2f} ± {time_std_12_ral:.2f}')

# time_mean_18_ral = np.mean(np.mean(timeHistory_18_ral, axis=0), axis=0)
# time_std_18_ral = np.mean(np.std(timeHistory_18_ral, axis=0))
# print(f'Time 18 RAL: {time_mean_18_ral[-1]:.2f} ± {time_std_18_ral:.2f}')

# time_mean_24_ral = np.mean(np.mean(timeHistory_24_ral, axis=0), axis=0)
# time_std_24_ral = np.mean(np.std(timeHistory_24_ral, axis=0))
# print(f'Time 24 RAL: {time_mean_24_ral[-1]:.2f} ± {time_std_24_ral:.2f}')

# RAL Data

path_gen = Path('6robs_ral')
sims_ral = ['sim-1', 'sim-2', 'sim-3']
rmseHistory_ral = []
nlpdHistory_ral = []
observedHistory_ral = []
filteredHistory_ral = []
DEC_gapx_time_hist_ral = []
DAC_time_hist_ral = []
timeHistory_ral = []
meanHistory_ral = []
stdHistory_ral = []

for sim in sims_ral:
    path = path_gen / sim
    robotHistory = np.load(path / "robotHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    observedHistory = np.load(path / "observedHistory.npy")
    filteredHistory = np.load(path / "filteredHistory.npy")
    DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
    DAC_time_hist = np.load(path / "DAC_time_hist.npy")
    timeHistory = np.load(path / "timeHistory.npy")
    meanHistory = np.load(path / "meanHistory.npy")
    stdHistory = np.load(path / "stdHistory.npy")

    rmseHistory_ral.append(rmseHistory)
    nlpdHistory_ral.append(nlpdHistory)
    observedHistory_ral.append(observedHistory)
    filteredHistory_ral.append(filteredHistory)
    DEC_gapx_time_hist_ral.append(DEC_gapx_time_hist)
    DAC_time_hist_ral.append(DAC_time_hist)
    timeHistory_ral.append(timeHistory)
    meanHistory_ral.append(meanHistory)
    stdHistory_ral.append(stdHistory)


# No Filter Data
path_gen = Path('proposed-no-nystrom')
sims_nofilter = ['sim-1', 'sim-2', 'sim-3']
timeHistory_nofilter = []
rmseHistory_nofilter = []
nlpdHistory_nofilter = []

for sim in sims_nofilter:
    path = path_gen / sim
    timeHistory = np.load(path / "timeHistory.npy")
    nlpdHistory = np.load(path / "nlpdHistory.npy")
    rmseHistory = np.load(path / "rmseHistory.npy")

    timeHistory_nofilter.append(timeHistory)
    nlpdHistory_nofilter.append(nlpdHistory)
    rmseHistory_nofilter.append(rmseHistory)


#######################
# ROBOT PLOTS
#######################
if general:
    selection = 3
    robotHistory = robotHistory_6[selection]
    meanHistory = meanHistory_6[selection]
    stdHistory = stdHistory_6[selection]
    fig = plt.figure(figsize=(10, 5))

    # # Inital position of the robots
    # ax = fig.add_subplot(121, aspect='equal')
    # c = ax.contourf(np.linspace(0, target_size[1], target_size[1]),
    #                 np.linspace(0, target_size[0], target_size[0]),
    #                 field, cmap=cmap, alpha=alpha, extend='both')
    # ax.set_xticks(np.arange(0, target_size[1] + 1, 10))
    # ax.set_yticks(np.arange(0, target_size[0] + 1, 10))
    # # Set fontdict for ticks
    # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label.set_fontname('serif')
    #     label.set_fontsize(fontsize)
    #     label.set_color('black')
    # # Plot the robot paths
    # for i in range(robNum):
    #     ax.scatter(robotHistory[i, 0, 0], robotHistory[i, 1, 0], marker='o', label=f'Robot {i}')
    #     ax.text(robotHistory[i, 0, 0] + 1, robotHistory[i, 1, 0] + 1, f'{i}', ha='center', va='center', fontdict=fontdict)
    # # Plot the voronoi regions
    # limRegions = utils.voronoi_alg_limited(robotHistory[:, :, 0], BBOX, sensRange)
    # for i, region in enumerate(limRegions):
    #     # Extract the exterior coordinates of the Polygon
    #     x, y = region.exterior.xy
    #     ax.plot(x, y, color="black", linewidth=2)
    # ax.grid(True, alpha=0.3)
    # ax.set_title('Time Step: 0', fontdict=fontdict)

    # divider = utils.make_axes_locatable(ax)
    # cax = divider.append_axes("bottom", size="5%", pad=0.5)
    # cbar = plt.colorbar(c, cax=cax, orientation="horizontal", format="%.1f")
    # cbar.ax.tick_params(rotation=90)
    # for tick in cbar.ax.get_xticklabels():
    #     tick.set_fontname('serif')
    #     tick.set_fontsize(fontsize)
    #     tick.set_color('black')
    # cbar.update_ticks()


    ax = fig.add_subplot(131, aspect='equal')
    c = ax.contourf(np.linspace(0, target_size[1], target_size[1]),
                np.linspace(0, target_size[0], target_size[0]),
                field, cmap=cmap, alpha=alpha, extend='both')
    ax.set_xticks(np.arange(0, target_size[1] + 1, 10))
    ax.set_yticks(np.arange(0, target_size[0] + 1, 10))
    # Set fontdict for ticks
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('serif')
        label.set_fontsize(fontsize)
        label.set_color('black')
    # Plot the robot paths
    for i in range(robNum):
        ax.plot(robotHistory[i, 0, :], robotHistory[i, 1, :])
        ax.scatter(robotHistory[i, 0, 0], robotHistory[i, 1, 0], s=80, facecolors='none', edgecolors=f'C{i}')
        ax.scatter(robotHistory[i, 0, -1], robotHistory[i, 1, -1], marker='o', label=f'Robot {i}')
        ax.text(robotHistory[i, 0, -1] + 1, robotHistory[i, 1, -1] + 1, f'{i}', ha='center', va='center', fontdict=fontdict)
    # Plot the voronoi regions
    limRegions = utils.voronoi_alg_limited(robotHistory[:, :, -1], BBOX, sensRange)
    for i, region in enumerate(limRegions):
        # Extract the exterior coordinates of the Polygon
        x, y = region.exterior.xy
        ax.plot(x, y, color="black", linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.set_title('Robots & Field', fontdict=fontdict)
    # colorbar
    divider = utils.make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = plt.colorbar(c, cax=cax, orientation="horizontal", format="%.1f")
    cbar.ax.tick_params(rotation=90)
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontname('serif')
        tick.set_fontsize(fontsize)
        tick.set_color('black')
    cbar.update_ticks()


    # Plot the Prediction Mean
    ax = fig.add_subplot(132, aspect='equal')
    c = ax.contourf(np.linspace(0, target_size[1], target_size[1]),
                    np.linspace(0, target_size[0], target_size[0]),
                    meanHistory[-1, 0, :, :], cmap=cmap, alpha=alpha, extend='both')
    ax.set_xticks(np.arange(0, target_size[1] + 1, 10))
    ax.set_yticks(np.arange(0, target_size[0] + 1, 10))
    ax.grid(True, alpha=0.3)
    ax.set_title('Prediction Mean', fontdict=fontdict)
    # Set fontdict for ticks
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('serif')
        label.set_fontsize(fontsize)
        label.set_color('black')
    # colorbar
    divider = utils.make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = plt.colorbar(c, cax=cax, orientation="horizontal", format="%.1f")
    cbar.ax.tick_params(rotation=90)
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontname('serif')
        tick.set_fontsize(fontsize)
        tick.set_color('black')
    cbar.update_ticks()


    # Plot the Prediction Std
    ax = fig.add_subplot(133, aspect='equal')
    c = ax.contourf(np.linspace(0, target_size[1], target_size[1]),
                    np.linspace(0, target_size[0], target_size[0]),
                    stdHistory[-1, 0, :, :], cmap='gray', alpha=alpha, extend='both')
    ax.set_xticks(np.arange(0, target_size[1] + 1, 10))
    ax.set_yticks(np.arange(0, target_size[0] + 1, 10))
    ax.grid(True, alpha=0.3)
    ax.set_title('Prediction Std.', fontdict=fontdict)
    # Set fontdict for ticks
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('serif')
        label.set_fontsize(fontsize)
        label.set_color('black')
    # colorbar
    divider = utils.make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = plt.colorbar(c, cax=cax, orientation="horizontal", format="%.1f")
    cbar.ax.tick_params(rotation=90)
    for tick in cbar.ax.get_xticklabels():
        tick.set_fontname('serif')
        tick.set_fontsize(fontsize)
        tick.set_color('black')
    cbar.update_ticks()
    plt.tight_layout()
    # Set general title
    fig.suptitle("6-Robots Team Simulation", weight='bold', size=18, color='black', family='serif')

    # plt.savefig('figures/6Robs_3subfigs_simulation.pdf', format='pdf', bbox_inches='tight', dpi=300)
    plt.show()



#######################
# METRICS PLOTS
#######################
if metrics:
    # # Plot the RMSE mean    
    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(121)
    # rmse_mean = np.mean(rmseHistory_6, axis=0)
    # rmse_std = np.std(rmseHistory_6, axis=0)
    # ax.plot(np.mean(rmse_mean, axis=0), label='Mean RMSE')
    # ax.fill_between(np.arange(rmse_mean.shape[1]), rmse_mean[0] - rmse_std[0], rmse_mean[0] + rmse_std[0], alpha=0.3)
    # ax.set_xlabel('Time')
    # ax.set_ylabel('RMSE')
    # ax.set_title('RMSE')
    # ax.legend()
    # ax.grid(alpha=0.3)

    # # # Plot the NLPD mean
    # ax = fig.add_subplot(122)
    # nlpd_mean = np.mean(nlpdHistory_6, axis=0)
    # nlpd_std = np.std(nlpdHistory_6, axis=0)
    # ax.plot(np.mean(nlpd_mean, axis=0), label='Mean NLPD')
    # ax.fill_between(np.arange(nlpd_mean.shape[1]), nlpd_mean[0] - nlpd_std[0], nlpd_mean[0] + nlpd_std[0], alpha=0.3)
    # ax.set_xlabel('Time')
    # ax.set_ylabel('NLPD')
    # ax.set_title('NLPD')
    # ax.legend()
    # ax.grid(alpha=0.3)

    # plt.show()

    # Plot the observed and filtered data
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    filtered_mean = np.mean(np.mean(filteredHistory_6, axis=0), axis=0)
    filtered_std = np.std(np.mean(filteredHistory_6, axis=0), axis=0)
    ax.plot(filtered_mean, label='Proposed')
    ax.fill_between(np.arange(filtered_mean.shape[0]), filtered_mean - filtered_std, filtered_mean + filtered_std, alpha=0.3)
    observe_mean = np.mean(np.mean(observedHistory_6, axis=0), axis=0)
    observe_std = np.std(np.mean(observedHistory_6, axis=0), axis=0)
    ax.plot(observe_mean, label='Baseline [12] w/o filtering',linestyle='--', alpha=0.5)
    ax.fill_between(np.arange(observe_mean.shape[0]), observe_mean - observe_std, observe_mean + observe_std, alpha=0.3)

    ax.set_xlabel('Time Steps', fontdict=fontdict)
    ax.set_ylabel('No. of Samples', fontdict=fontdict)
    ax.set_xticks(np.arange(0, filtered_mean.shape[0] + 1, 25))
    ax.legend(fontsize=fontsize, loc='upper left')
    ax.set_ylim(0, 180)
    # Percentage of samples not used in FL
    perc = 100 * (1 - (filtered_mean[-1] / observe_mean[-1]))
    # Fill between the two curves
    ax.fill_between(np.arange(observe_mean.shape[0]), observe_mean, filtered_mean, where=filtered_mean < observe_mean, color='green', alpha=0.1, zorder=0)
    # Create text box
    ax.text(0.75, 0.5, f'{perc:.2f}%\nSample Reduction', 
            transform=ax.transAxes, 
            fontsize=fontsize, 
            verticalalignment='center', 
            horizontalalignment='center', 
            color='green', 
            weight='bold',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    # Create context box
    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('serif')
        label.set_fontsize(fontsize)
        label.set_color('black')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/observed_filtered_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.show()

    # # # Plot the DEC gapx time history
    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(111)
    # DEC_gapx_time_hist_mean = np.mean(DEC_gapx_time_hist_6, axis=0).squeeze()
    # DEC_gapx_time_hist_std = np.std(DEC_gapx_time_hist_6, axis=0).squeeze()
    # ax.plot(DEC_gapx_time_hist_mean, label='Mean DEC gapx')
    # ax.fill_between(np.arange(DEC_gapx_time_hist_mean.shape[0]), DEC_gapx_time_hist_mean - DEC_gapx_time_hist_std, DEC_gapx_time_hist_mean + DEC_gapx_time_hist_std, alpha=0.3)
    # DAC_time_hist_mean = np.mean(DAC_time_hist_6, axis=0).squeeze()
    # DAC_time_hist_std = np.std(DAC_time_hist_6, axis=0).squeeze()
    # ax.plot(DAC_time_hist_mean, label='Mean DAC')
    # ax.fill_between(np.arange(DAC_time_hist_mean.shape[0]), DAC_time_hist_mean - DAC_time_hist_std, DAC_time_hist_mean + DAC_time_hist_std, alpha=0.3)
    # timeHistory_mean = np.mean(timeHistory_6, axis=0).squeeze()
    # timeHistory_std = np.std(timeHistory_6, axis=0).squeeze()
    # ax.plot(timeHistory_mean, label='Mean Time')
    # ax.fill_between(np.arange(timeHistory_mean.shape[0]), timeHistory_mean - timeHistory_std, timeHistory_mean + timeHistory_std, alpha=0.3)
    # ax.set_xlabel('Period')
    # ax.set_ylabel('Time')
    # ax.set_title('Time history')
    # ax.legend()
    # ax.grid(alpha=0.3)

    # # Plot the time history
    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(111)
    # time_mean = np.mean(timeHistory_6, axis=0).squeeze()
    # time_std = np.std(timeHistory_6, axis=0).squeeze()
    # ax.plot(time_mean, label='Proposed')
    # ax.fill_between(np.arange(time_mean.shape[0]), time_mean - time_std, time_mean + time_std, alpha=0.3)
    # time_mean_nofilter = np.mean(timeHistory_nofilter, axis=0).squeeze()
    # time_std_nofilter = np.std(timeHistory_nofilter, axis=0).squeeze()
    # ax.plot(time_mean_nofilter, label='Baseline [12] w/o filtering', linestyle='--')
    # ax.fill_between(np.arange(time_mean_nofilter.shape[0]), time_mean_nofilter - time_std_nofilter, time_mean_nofilter + time_std_nofilter, alpha=0.3)
    # # ax.plot(timeHistory_mean_nofilter, label='Mean Time (No Filter)', linestyle='--')
    # # ax.fill_between(np.arange(timeHistory_mean_nofilter.shape[1]), timeHistory_mean_nofilter[0] - timeHistory_std_nofilter[0], timeHistory_mean_nofilter[0] + timeHistory_std_nofilter[0], alpha=0.3)
    # ax.set_xlabel('Time Steps', fontdict=fontdict)
    # ax.set_ylabel('Comp. Time [s]', fontdict=fontdict)
    # ax.set_xticks(np.arange(0, time_mean_nofilter.shape[0] + 1, 25))
    # ax.legend(fontsize=fontsize, loc='upper left')
    # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    #     label.set_fontname('serif')
    #     label.set_fontsize(fontsize)
    #     label.set_color('black')
    # ax.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig('figures/time_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300, pad_inches=0)

    # Plot the RMSE mean
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    rmse_mean = np.mean(rmseHistory_6, axis=0)
    rmse_std = np.std(rmseHistory_6, axis=0)
    ax.plot(np.mean(rmse_mean, axis=0), label='Proposed')
    ax.fill_between(np.arange(rmse_mean.shape[1]), rmse_mean[0] - rmse_std[0], rmse_mean[0] + rmse_std[0], alpha=0.3)
    rmse_mean_nofilter = np.mean(rmseHistory_nofilter, axis=0)
    rmse_std_nofilter = np.std(rmseHistory_nofilter, axis=0)
    ax.plot(np.mean(rmse_mean_nofilter, axis=0), label='Baseline [12] w/o filtering', linestyle='--')
    ax.fill_between(np.arange(rmse_mean_nofilter.shape[1]), rmse_mean_nofilter[0] - rmse_std_nofilter[0], rmse_mean_nofilter[0] + rmse_std_nofilter[0], alpha=0.3)
    ax.set_xlabel('Time Steps', fontdict=fontdict)
    ax.set_ylabel('RMSE', fontdict=fontdict)
    ax.set_xticks(np.arange(0, len(rmse_mean_nofilter[0]) + 1, 25))
    ax.legend(fontsize=fontsize, loc='upper left')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('serif')
        label.set_fontsize(fontsize)
        label.set_color('black')
    ax.grid(alpha=0.3)

    # Plot the NLPD mean
    ax = fig.add_subplot(122)
    nlpd_mean = np.mean(nlpdHistory_6, axis=0)
    nlpd_std = np.std(nlpdHistory_6, axis=0)
    ax.plot(np.mean(nlpd_mean, axis=0), label='Proposed')
    ax.fill_between(np.arange(nlpd_mean.shape[1]), nlpd_mean[0] - nlpd_std[0], nlpd_mean[0] + nlpd_std[0], alpha=0.3)
    nlpd_mean_nofilter = np.mean(nlpdHistory_nofilter, axis=0)
    nlpd_std_nofilter = np.std(nlpdHistory_nofilter, axis=0)
    ax.plot(np.mean(nlpd_mean_nofilter, axis=0), label='Baseline [12] w/o filtering', linestyle='--')
    ax.fill_between(np.arange(nlpd_mean_nofilter.shape[1]), nlpd_mean_nofilter[0] - nlpd_std_nofilter[0], nlpd_mean_nofilter[0] + nlpd_std_nofilter[0], alpha=0.3)
    ax.set_xlabel('Time Steps', fontdict=fontdict)
    ax.set_ylabel('NLPD', fontdict=fontdict)
    ax.set_xticks(np.arange(0, len(nlpd_mean_nofilter[0]) + 1, 25))
    ax.legend(fontsize=fontsize, loc='upper left')
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('serif')
        label.set_fontsize(fontsize)
        label.set_color('black')
    ax.grid(alpha=0.3)
    # fig.suptitle("Proposed Vs No Filter", weight='bold', size=18, color='black', family='serif', y=1)
    # Increare padding between subplots

    plt.tight_layout()
    plt.savefig('figures/metrics_comparison.pdf', format='pdf', bbox_inches='tight', dpi=300, pad_inches=0)
    plt.show()



######################
# TIME COMPARISON
######################

# Time comparison with RAL
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
time_mean = np.mean(timeHistory_6, axis=0).squeeze()
time_std = np.std(timeHistory_6, axis=0).squeeze()
ax.plot(time_mean, label='Mean Time Proposed')
ax.fill_between(np.arange(time_mean.shape[0]), time_mean - time_std, time_mean + time_std, alpha=0.3)
time_mean_ral = np.mean(timeHistory_ral, axis=0).squeeze()
time_std_ral = np.std(timeHistory_ral, axis=0).squeeze()
ax.plot(time_mean_ral, label='Mean Time RAL', linestyle='--')
ax.fill_between(np.arange(time_mean_ral.shape[0]), time_mean_ral - time_std_ral, time_mean_ral + time_std_ral, alpha=0.3)
ax.set_xlabel('Period')
ax.set_ylabel('Comp. Time')
ax.set_title('Time comparison')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# robotHistory = np.load(path / "robotHistory.npy")
# rmseHistory = np.load(path / "rmseHistory.npy")
# nlpdHistory = np.load(path / "nlpdHistory.npy")
# observedHistory = np.load(path / "observedHistory.npy")
# filteredHistory = np.load(path / "filteredHistory.npy")
# DEC_gapx_time_hist = np.load(path / "DEC_gapx_time_hist.npy")
# DAC_time_hist = np.load(path / "DAC_time_hist.npy")
# timeHistory = np.load(path / "timeHistory.npy")
# meanHistory = np.load(path / "meanHistory.npy")
# stdHistory = np.load(path / "stdHistory.npy")

# PERIOD = robotHistory.shape[2]
# coverageHistory = np.empty((PERIOD, 1))

# for t in range(PERIOD):
#     print(f"Time: {t}")
#     coverageHistory[t] = utils.coveragePerformanceFuncDataset(robotHistory[:, :, t], field, res=10, BBOX=BBOX)

# Plot the data
# Only sim 0



# ax.legend()

# # Plot the RMSE
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# for i in range(robNum):
#     ax.plot(rmseHistory[i, :], label=f'Robot {i}')
# ax.set_xlabel('Time')
# ax.set_ylabel('RMSE')
# ax.set_title('RMSE')
# ax.legend()

# # Plot the NLPD
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# for i in range(robNum):
#     ax.plot(nlpdHistory[i, :], label=f'Robot {i}')
# ax.set_xlabel('Time')
# ax.set_ylabel('NLPD')
# ax.set_title('NLPD')
# ax.legend()

# # Plot the coverage performance function
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# ax.plot(coverageHistory, label='Coverage')
# ax.set_xlabel('Time')
# ax.set_ylabel('Coverage')
# ax.set_title('Coverage')
# ax.legend()

# # Plot the observed and filtered data
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# for i in range(robNum):
#     ax.plot(observedHistory[i, :], label=f'Robot {i} observed', alpha=0.5, c=f'C{i}')
# for i in range(robNum):
#     ax.plot(filteredHistory[i, :], label=f'Robot {i} filtered', linestyle='--',  c=f'C{i}')
# ax.set_xlabel('Time')
# ax.set_ylabel('Data')
# ax.set_title('Observed and filtered data')
# ax.legend()

# # Plot the DEC gapx time history
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(111)
# ax.plot(DEC_gapx_time_hist, label='DEC gapx')
# # Fill the area under the curve
# ax.fill_between(np.arange(PERIOD), DEC_gapx_time_hist[:, 0], alpha=0.3)
# ax.plot(DAC_time_hist, label='DAC')
# ax.fill_between(np.arange(PERIOD), DAC_time_hist[:, 0], alpha=0.5)
# ax.plot(timeHistory, label='Time')
# ax.fill_between(np.arange(PERIOD), timeHistory[:, 0], alpha=0.2)
# ax.set_xlabel('Time')
# ax.set_ylabel('Time')
# ax.set_title('Time history')
# ax.legend()

# plt.show()
