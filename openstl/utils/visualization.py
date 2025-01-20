from typing import Union

import cv2
import os
import imageio
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


def imshow(img: Union[str, np.ndarray],
           win_name: str = '',
           wait_time: int = 0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    assert isinstance(img, np.ndarray)
    cv2.imshow(win_name, img)
    if wait_time == 0:  # prevent from hanging if windows was closed
        while True:
            ret = cv2.waitKey(1)
            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)


def plot_world_map(out_path=None):
    """plot the world map with Basemap"""
    try:
        from mpl_toolkits.basemap import Basemap
    except:
        assert False and 'Please install Basemap, e.g., pip install geos basemap pyproj'

    fig = plt.figure(figsize=(8, 4))
    fig.add_axes([0., 0., 1, 1])
    map = Basemap()
    map.drawcoastlines(linewidth=2)
    map.drawcountries(linewidth=1)
    plt.show()
    if out_path is not None:
        plt.savefig(out_path, dpi=300, format='png')
    plt.close()


def get_mpl_colormap(cmap_name):
    """mapping matplotlib cmap to cv2"""
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:,2::-1]

    return color_range.reshape(256, 1, 3)


def show_video_line(data, ncols, vmax=0.6, vmin=0.0, cmap='gray', norm=None, cbar=False, format='png', out_path=None,
                    pred_length=None, use_rgb=False):
    """generate images with a video sequence"""
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(3.25 * ncols, 3))
    plt.subplots_adjust(wspace=0.01, hspace=0)

    if len(data.shape) > 3:
        data = data.swapaxes(1,2).swapaxes(2,3)

    images = []
    if ncols == 1:
        if use_rgb:
            im = axes.imshow(cv2.cvtColor(data[0], cv2.COLOR_BGR2RGB))
        else:
            im = axes.imshow(data[0], cmap=cmap, norm=norm)
        images.append(im)
        axes.axis('off')
        im.set_clim(vmin, vmax)
    else:
        if pred_length is not None:
            input_length = ncols - pred_length
            texts = ["input"] * input_length + ["pred"] * pred_length
        for t, ax in enumerate(axes.flat):
            if use_rgb:
                im = ax.imshow(cv2.cvtColor(data[t], cv2.COLOR_BGR2RGB), cmap='gray')
            else:
                im = ax.imshow(data[t], cmap=cmap, norm=norm)
            images.append(im)
            ax.axis('off')
            im.set_clim(vmin, vmax)
            if pred_length is not None:
                ax.text(0.5, 1.10, texts[t], color='black', fontsize=12, ha='center', va='top', transform=ax.transAxes)

    if cbar and ncols > 1:
        cbaxes = fig.add_axes([0.9, 0.15, 0.04 / ncols, 0.7])
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.1, cax=cbaxes)

    plt.show()
    if out_path is not None:
        fig.savefig(out_path, format=format, pad_inches=0, bbox_inches='tight')
    plt.close()


def show_video_gif_multiple(prev, true, pred, vmax=0.6, vmin=0.0, cmap='gray', norm=None, out_path=None, use_rgb=False):
    """generate gif with a video sequence"""

    def swap_axes(x):
        if len(x.shape) > 3:
            return x.swapaxes(1,2).swapaxes(2,3)
        else: return x

    prev, true, pred = map(swap_axes, [prev, true, pred])
    prev_frames = prev.shape[0]
    frames = prev_frames + true.shape[0]
    images = []
    for i in range(frames):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))
        for t, ax in enumerate(axes):
            if t == 0:
                plt.text(0.3, 1.05, 'ground truth', fontsize=15, color='green', transform=ax.transAxes)
                if i < prev_frames:
                    if use_rgb:
                        im = ax.imshow(cv2.cvtColor(prev[i], cv2.COLOR_BGR2RGB))
                    else:
                        im = ax.imshow(prev[i], cmap=cmap, norm=norm)
                else:
                    if use_rgb:
                        im = ax.imshow(cv2.cvtColor(true[i-frames], cv2.COLOR_BGR2RGB))
                    else:
                        im = ax.imshow(true[i-frames], cmap=cmap, norm=norm)
            elif t == 1:
                plt.text(0.2, 1.05, 'predicted frames', fontsize=15, color='red', transform=ax.transAxes)
                if i < prev_frames:
                    if use_rgb:
                        im = ax.imshow(cv2.cvtColor(prev[i], cv2.COLOR_BGR2RGB))
                    else:
                        im = ax.imshow(prev[i], cmap=cmap, norm=norm)
                else:
                    if use_rgb:
                        im = ax.imshow(cv2.cvtColor(pred[i-frames], cv2.COLOR_BGR2RGB))
                    else:
                        im = ax.imshow(pred[i-frames], cmap=cmap, norm=norm)
            ax.axis('off')
            im.set_clim(vmin, vmax)
        plt.savefig('./tmp.png', bbox_inches='tight', format='png')
        images.append(Image.fromarray(np.asarray(Image.open('./tmp.png'))))
    plt.close()
    os.remove('./tmp.png')

    fps = 10

    if out_path is not None:
        if not out_path.endswith('gif'):
            out_path = out_path + '.gif'
        images[0].save(out_path, save_all=True, append_images=images[1:], optimize=False, duration=int(1/fps*len(images)), loop=0)
    return images


def show_video_mp4_multiple(prev_list, true_list, pred_list,
                            vmax=0.6, vmin=0.0, cmap='gray', norm=None,
                            out_path=None, use_rgb=False):
    """Generate an MP4 video with a video sequence of (prev + true) vs (prev + pred)."""

    # Utility to swap axes if needed (as in your original code).
    def swap_axes(x):
        if len(x.shape) > 3:
            # (batch, height, width, channels) -> (batch, width, height, channels), etc.
            return x.swapaxes(1, 2).swapaxes(2, 3)
        else:
            return x

    video_writer = None  # Will initialize after first frame size is known
    frames = None
    for index in range(len(pred_list)):
        prev = prev_list[index]
        true = true_list[index]
        pred = pred_list[index]

        prev, true, pred = map(swap_axes, [prev, true, pred])
        prev_frames = prev.shape[0]
        frames = prev_frames + true.shape[0]

        # If an output path is specified, ensure it ends with .mp4
        if out_path is not None:
            if not out_path.endswith('.mp4'):
                out_path = out_path + '.mp4'

        fps = 10  # frames per second for the video

        for i in range(frames):
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 6))

            for t, ax in enumerate(axes):
                if t == 0:
                    ax.set_title('Ground Truth', fontsize=15, color='green')
                    # Show either prev or true
                    if i < prev_frames:
                        # Show a frame from prev
                        frame_data = prev[i]
                    else:
                        # Show a frame from true
                        frame_data = true[i - prev_frames]
                else:
                    ax.set_title('Predicted Frames', fontsize=15, color='red')
                    # Show either prev or pred
                    if i < prev_frames:
                        frame_data = prev[i]
                    else:
                        frame_data = pred[i - prev_frames]

                # If using RGB images, convert from BGR to RGB for proper Matplotlib display
                if use_rgb:
                    # Already presumably in BGR, so convert to RGB for imshow
                    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)
                    im = ax.imshow(frame_data)
                else:
                    im = ax.imshow(frame_data, cmap=cmap, norm=norm)

                # Set color limits
                im.set_clim(vmin, vmax)
                ax.axis('off')

            # Draw the figure so we can grab the pixel buffer
            fig.canvas.draw()

            # Convert the canvas to an RGB image (height x width x 3)
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # Initialize the video writer once we know the frame dimensions
            if video_writer is None and out_path is not None:
                height, width, _ = frame.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'avc1', 'mp4v', or 'X264' can work
                video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            # Matplotlib gives us RGB; OpenCV writes in BGR by default
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if video_writer is not None:
                video_writer.write(frame_bgr)

            plt.close(fig)  # Close the figure to avoid memory buildup

    # Release the writer
    if video_writer is not None:
        video_writer.release()

    # Optionally return something if needed, e.g. return frames, or None
    return frames

def show_video_gif_single(data, out_path=None, use_rgb=False):
    """generate gif with a video sequence"""
    images = []
    if len(data.shape) > 3:
        data=data.swapaxes(1, 2).swapaxes(2, 3)

    images = []
    for i in range(data.shape[0]):
        if use_rgb:
            data[i] = cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB)
        image = imageio.core.util.Array(data[i])
        # if the channel is 1, need to reduce this channel dimension
        if image.shape[2] == 1:
            image = np.squeeze(image, axis=2)
        image = (image * 255).astype(np.uint8)
        images.append(image)

    if out_path is not None:
        if not out_path.endswith('gif'):
            out_path = out_path + '.gif'
        imageio.mimsave(out_path, images)


def show_heatmap_on_image(img: np.ndarray,
                          mask: np.ndarray,
                          use_rgb: bool = False,
                          colormap: int = cv2.COLORMAP_JET,
                          image_weight: float = 0.5,
                          image_binary: bool = False) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

        img: The base image in RGB or BGR format.
        mask: The cam mask.
        use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        colormap: The OpenCV colormap to be used.
        image_weight: The final result is image_weight * img + (1-image_weight) * mask.
        image_binary: Whether to binarize the image.

    returns: The default image with the cam overlay.
    """
    if mask.shape[0] != img.shape[1]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1]. Got: {image_weight}")

    if not image_binary:
        cam = (1 - image_weight) * heatmap + image_weight * img
    else:
        cam = (1 - image_weight) * heatmap + image_weight * img
        mask = 255 * img[:, :, 0] < 100.
        cam[mask, :] = img[mask, :]
    cam = cam / np.max(cam)

    return np.uint8(255 * cam)


def show_taxibj(heatmap, cmap='viridis', title=None, out_path=None, vis_channel=None):
    """ploting heatmap to show or save of TaxiBJ"""
    if vis_channel is not None:
        vis_channel = 0 if vis_channel < 0 else vis_channel
    else:
        vis_channel = 0

    cmap = get_mpl_colormap(cmap)
    ret_img = list()
    if len(heatmap.shape) == 3:
        heatmap = heatmap[np.newaxis, :]

    for i in range(heatmap.shape[0]):
        # plot heatmap with cmap
        vis_img = heatmap[i, vis_channel, :, :, np.newaxis]
        vis_img = cv2.resize(np.uint8(255 * vis_img), (256, 256)).squeeze()
        vis_img = cv2.applyColorMap(np.uint8(vis_img), cmap)
        vis_img = np.float32(vis_img) / 255
        vis_img = vis_img / np.max(vis_img)
        vis_img = np.uint8(255 * vis_img)

        ret_img.append(vis_img[np.newaxis, :])
        if out_path is not None:
            cv2.imwrite(str(out_path).replace('.', f'{i}.'), vis_img)
        if title is not None:
            imshow(vis_img, win_name=title+str(i))

    if len(ret_img) > 1:
        return np.concatenate(ret_img, axis=0)
    else:
        return ret_img[0]


def show_weather_bench(heatmap, src_img=None, cmap='GnBu', title=None,
                       out_path=None, vis_channel=None):
    """fusing src_img and heatmap to show or save of Weather Bench"""
    if not isinstance(src_img, np.ndarray):
        if src_img is None:
            plot_world_map('tmp.png')
            src_img = cv2.imread('tmp.png')
            os.remove('./tmp.png')
        elif isinstance(src_img, str):
            src_img = cv2.imread(src_img)
        src_img = cv2.resize(src_img, (512, 256))
    src_img = np.float32(src_img) / 255
    if vis_channel is not None:
        vis_channel = 0 if vis_channel < 0 else vis_channel
    else:
        vis_channel = 0

    ret_img = list()
    if len(heatmap.shape) == 3:
        heatmap = heatmap[np.newaxis, :]

    for i in range(heatmap.shape[0]):
        vis_img = show_heatmap_on_image(
            src_img, heatmap[i, vis_channel, ...], use_rgb=False, colormap=get_mpl_colormap(cmap),
            image_weight=0.1, image_binary=True)
        ret_img.append(vis_img[np.newaxis, :])
        if out_path is not None:
            cv2.imwrite(str(out_path).replace('.', f'{i}.'), vis_img)
        if title is not None:
            imshow(vis_img, win_name=title+str(i))

    if len(ret_img) > 1:
        return np.concatenate(ret_img, axis=0)
    else:
        return ret_img[0]
