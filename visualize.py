import matplotlib.pyplot as plt
import numpy as np

def visualize_3d_slice(img: np.array, fig_ax: plt.Axes, title: str = ""):
    """ 
    Takes in a 3d image of shape (depth, height, width) and plots each z-slice as a row of 2D images on the given axis.
    
    Args:
        img (np.array): 3D image to visualize (depth, height, width)
        fig_ax (plt.Axes): matplotlib axis to plot on
        title (str): title of the plot
        
    Sample Usage:
        fig, ax = plt.subplots(3, depth, figsize=(15, 5), num=1)
        visualize_3d_slice(input_img, ax[0], "Input")
        visualize_3d_slice(label_img, ax[1], "Ground Truth")
        visualize_3d_slice(pred_img, ax[2], "Prediction")
    """
    depth, width, height = img.shape
    for i in range(depth):
        fig_ax[i].imshow(img[i], cmap="gray")
    fig_ax[0].set_ylabel(title)