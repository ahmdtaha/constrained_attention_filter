import imageio
import matplotlib
matplotlib.use('Agg')
import numpy as np
import constants as const
from skimage import transform
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def save_tight(filepath):
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)

    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(ticker.NullLocator())
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)

def apply_heatmap(image, heatmap, alpha=0.6, display=False, save=None, cmap='viridis', axis='on', verbose=False):

    height = image.shape[0]
    width = image.shape[1]

    # resize heat map
    heat_map_resized = transform.resize(heatmap, (height, width))

    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    # display
    plt.imshow(image)
    plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    plt.axis(axis)

    if display:
        plt.show()

    if save is not None:
        if verbose:
            print('save image: ' + save)
        # plt.savefig(save, bbox_inches='tight', pad_inches=0)
        save_tight(save)

    return plt

def save_heatmap(heatmap,save=None):
    heat_map_resized = transform.resize(heatmap, (const.max_frame_size, const.max_frame_size))
    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = 255* (heat_map_resized - min_value) / (max_value - min_value)

    imageio.imwrite(save,normalized_heat_map)
