
import os
from PIL import Image
import moviepy.video.io.ImageSequenceClip

def make_frames_side_by_side(path1, path2, output_path, save_images_side_by_side):
    r"""Create side-by-side frames from two directories of PNG images.

    Parameters
    ----------
    path1 : str
        Path to the first directory containing PNG images.
    path2 : str
        Path to the second directory containing PNG images.
    output_path : str
        Directory where the side-by-side frames will be saved.
    save_images_side_by_side : callable
        Function to save two images side by side into a single file.

    Returns
    -------
    frames_side_by_side : str
        Path to the directory containing the generated side-by-side frames.
    """
    files1 = sorted([f for f in os.listdir(path1) if f.endswith('.png')])
    files2 = sorted([f for f in os.listdir(path2) if f.endswith('.png')])
    frames_side_by_side = os.path.join(output_path, 'frames_side_by_side')
    os.makedirs(frames_side_by_side, exist_ok=True)
    for file1, file2 in zip(files1, files2):
        frame_idx = file1.split('_')[-1]
        out_file = os.path.join(frames_side_by_side, f'side_by_side_{frame_idx}')
        save_images_side_by_side(
            os.path.join(path1, file1),
            os.path.join(path2, file2),
            out_file
        )
    return frames_side_by_side

def make_video(frames_path, fps=5, output_file=None):
    r"""Create a video from a sequence of PNG frames in a directory.

    Parameters
    ----------
    frames_path : str
        Path to the directory containing PNG frames.
    fps : int
        Frames per second for the output video. Default is 5.
    output_file : str
        Path to the output video file. If None, saves as 'video.mp4' in frames_path.

    Returns
    -------
    output_file : str
        Path to the generated video file.
    """
    files = os.listdir(frames_path)
    files = [os.path.join(frames_path, file) for file in files if os.path.isfile(os.path.join(frames_path, file)) and file.lower().endswith('.png')]
    files = sorted(files)
    if not files:
        raise RuntimeError('No frames found to make video.')
    files.insert(0, files[0])
    files.append(files[-1])
    output_file = output_file or os.path.join(frames_path, 'video.mp4')
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(files, fps=fps)
    clip.write_videofile(output_file)
    return output_file

def save_images_side_by_side(file1, file2, outfile):
    r"""Combine two images horizontally and save as a single image.

    Parameters
    ----------
    file1 : str
        Path to the first image file.
    file2 : str
        Path to the second image file.
    outfile : str
        Path to save the combined image.
    """
    img1 = Image.open(file1)
    img2 = Image.open(file2)
    total_width = img1.width + img2.width
    max_height = max(img1.height, img2.height)
    new_img = Image.new('RGB', (total_width, max_height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    new_img.save(outfile)
