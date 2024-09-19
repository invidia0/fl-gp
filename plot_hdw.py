import numpy as np
import utilities as utils
import matplotlib.pyplot as plt

def plot_mini_quadcopter(x, y, ax, color='black'):
    ax.scatter(x+4, y+4, s=100, marker='o', color=color, zorder=2)
    ax.scatter(x-4, y-4, s=100, marker='o', color=color, zorder=2)
    ax.scatter(x+4, y-4, s=100, marker='o', color=color, zorder=2)
    ax.scatter(x-4, y+4, s=100, marker='o', color=color, zorder=2)
    ax.plot([x, x+4], [y, y+4], color=color, zorder=2)
    ax.plot([x, x-4], [y, y-4], color=color, zorder=2)
    ax.plot([x, x+4], [y, y-4], color=color, zorder=2)
    ax.plot([x, x-4], [y, y+4], color=color, zorder=2)



# Generate data
np.random.seed(0)

area_size = 200
x_inf, y_inf = 0, 0
x_sup, y_sup = area_size, area_size
BBOX = [x_inf, y_inf, x_sup, y_sup]
d_field_ = 1
x1_ = np.arange(x_inf, x_sup + d_field_, d_field_)
x2_ = np.arange(y_inf, y_sup + d_field_, d_field_)
_X1, _X2 = np.meshgrid(x1_, x2_)
mesh = np.vstack([_X1.ravel(), _X2.ravel()]).T

# Generate random means
peaks = 4 # np.random.randint(1, 10)
means = np.random.uniform(low=0, high=area_size, size=(peaks, 2))
sigma = 30
Z = utils.gmm_pdf_array(mesh[:, 0], mesh[:, 1], sigma, means, flag_normalize=False)
Z = Z.reshape(len(x1_), len(x2_))
field = Z

""" Robots parameters """
ROB_NUM = 6
CAMERA_BOX = 10
CAMERA_SAMPLES = 10

_area_to_cover = (x_sup * y_sup) * 2.0
RANGE = 2 * np.sqrt((_area_to_cover / ROB_NUM) / np.pi)

fontsize = 18
fontdict = {'weight': 'bold', 
            'size': fontsize, 
            'color': 'black',
            'family': 'serif'}
alpha = 0.8
cmap = 'hot'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

robotHistory = np.load('robotHistory.npy')
end = 143

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(121)
# Load png image
img = plt.imread('figures/hdwPlot.png')
ax.imshow(img)
ax.set_aspect('equal')
# Remove grid lines
ax.grid(False)
# Remove labels
ax.set_xticklabels([])
ax.set_yticklabels([])
# Remove tick marks
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
# Remove frame
for spine in ax.spines.values():
    spine.set_visible(False)
# Remove title
ax.set_title('')


ax = fig.add_subplot(122)
ax.contourf(x1_, x2_, field, cmap=cmap, alpha=alpha)
ax.set_aspect('equal')
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('serif')
        label.set_fontsize(fontsize)
        label.set_color('black')
for i in range(ROB_NUM):
    ax.scatter(robotHistory[i, 0, 0], robotHistory[i, 1, 0], s=100, linewidths=2, facecolors='none', edgecolors=f'C{i}')
    # ax.scatter(robotHistory[i, 0, end], robotHistory[i, 1, end], s=100, marker='o')
    ax.plot(robotHistory[i, 0, :end], robotHistory[i, 1, :end], color=f'C{i}', alpha=1, linewidth=2, zorder=1)
    ax.text(robotHistory[i, 0, end] + 10, robotHistory[i, 1, end] + 10, f'{i+2}', fontdict=fontdict, ha='center', va='center')
    plot_mini_quadcopter(robotHistory[i, 0, end], robotHistory[i, 1, end], ax, color=f'C{i}')
limRegions = utils.voronoi_alg_limited(robotHistory[:, :, end], BBOX, RANGE)
for i, region in enumerate(limRegions):
    # Extract the exterior coordinates of the Polygon
    x, y = region.exterior.xy
    ax.plot(x, y, color="black", linewidth=2)
# Show the last tick
ax.set_xticks(np.arange(x_inf, x_sup + 1, 50))
ax.set_yticks(np.arange(y_inf, y_sup + 1, 50))
# Remove the tick labels
ax.set_xticklabels([])
ax.set_yticklabels([])
# Remove the tick marks
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
# Set the title with latex
# params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
# plt.rcParams.update(params)
# ax.set_title(f'Area: 400m\u00b2', fontdict=fontdict)
# ax.set_title('Quadrotors and Field to Cover', fontdict=fontdict)
ax.grid(True, alpha=0.5)
# plt.savefig('figures/hdw_quadrotors_field.pdf', bbox_inches='tight', dpi=300, pad_inches=0, format='pdf')
fig.suptitle(f'Area to cover: 400m\u00b2', fontsize=fontsize, fontweight='bold', fontfamily='serif', y=0.95)
# # Remove grid lines
# ax.grid(False)
# # Remove labels
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# # Remove tick marks
# ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
# # Remove frame
# for spine in ax.spines.values():
#     spine.set_visible(False)
# # Remove title
# ax.set_title('')
# # Remove x and y axis labels
# ax.set_xlabel('')
# ax.set_ylabel('')
# plt.tight_layout()
# # Remove white space around the image
# plt.margins(0)
# plt.savefig('figures/hdw_field_of_interest.png', bbox_inches='tight', dpi=300, pad_inches=0)
plt.savefig('figures/hdw_quadrotors_field.pdf', bbox_inches='tight', dpi=300, pad_inches=0, format='pdf')
plt.tight_layout()
plt.show()