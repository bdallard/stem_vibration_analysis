import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
import os
from scipy import fftpack



#################################
#
#
#     Helper functions
#
#
#################################



def save_frames(frames, folder_path):
    # If it doesn't exist, create the output folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(folder_path, f'frame_{i}.png'), frame)


#
# zoom on the steem 

def crop_video(input_video_path, output_video_path, top_percentage, right_percentage):
    """
    Crops a video based on the specified top and right percentage values.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to save the output cropped video.
        top_percentage (float): Percentage of the top region to keep (0-1).
        right_percentage (float): Percentage of the right region to keep (0-1).

    """

    cap = cv2.VideoCapture(input_video_path)

    # Check if the input video file is valid
    if not cap.isOpened():
        print(f"Failed to open the input video file: {input_video_path}")
        return

    # Get the original video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(width * right_percentage), int(height * top_percentage)))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[:int(height * top_percentage), :int(width * right_percentage)]

        # Write the cropped frame
        out.write(cropped_frame)

    # Release everything if the job is finished
    cap.release()
    out.release()


def extract_frames(video_path):
    """
    Extracts frames from a video and returns a list of frames.

    Args:
        video_path (str): Path to the video file.

    Returns:
        frames (list): List of frames.

    """

    cap = cv2.VideoCapture(video_path)
    frames = []

    # Check if the video file is valid
    if not cap.isOpened():
        print(f"Failed to open the video file: {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames
  

def display_saved_images(output_folder, interval=60):
    """
    Displays saved images from the specified output folder at a given interval.

    Args:
        output_folder (str): Path to the folder containing the saved images.
        interval (int, optional): The interval at which to display the images. Defaults to 60.

    """
    if not os.path.exists(output_folder):
        print(f"Output folder '{output_folder}' does not exist.")
        return
    
    saved_images = sorted(os.listdir(output_folder))

    for i, img_name in enumerate(saved_images):
        if i % interval == 0:
            # Read the image
            img = cv2.imread(img_path)
            # Convert color style from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Display the image
            plt.imshow(img)
            plt.title(img_name)
            plt.show()
    
    
def rgb_to_hsv(r, g, b):
    """
    Converts RGB values to HSV color space.

    Args:
        r (int): Red channel value (0-255).
        g (int): Green channel value (0-255).
        b (int): Blue channel value (0-255).

    Returns:
        hsv (numpy.ndarray): HSV values.

    """

    rgb = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv[0][0]


def create_hsv_bounds(h, s, v, tolerance=0.15):
    """
    Creates lower and upper bounds for HSV color filtering based on given HSV values and tolerance.

    Args:
        h (float): Hue value (0-360).
        s (float): Saturation value (0-1).
        v (float): Value (brightness) value (0-1).
        tolerance (float, optional): Tolerance level for adjusting the bounds. Defaults to 0.15.

    Returns:
        lower_bound (numpy.ndarray): Lower bound for HSV values.
        upper_bound (numpy.ndarray): Upper bound for HSV values.

    """

    lower_bound = np.array([h - (180 * tolerance), max(0, s - (255 * tolerance)), max(0, v - (255 * tolerance))])
    upper_bound = np.array([h + (180 * tolerance), min(255, s + (255 * tolerance)), min(255, v + (255 * tolerance))])
    return lower_bound, upper_bound


def plot_displacement(displacements, title):
    """
    Plots the displacement values over time.

    Args:
        displacements (list): List of displacement values.
        title (str): Title for the plot.

    """

    t = range(len(displacements))

    plt.figure(figsize=(10, 6))
    plt.plot(t, displacements, label='Displacement')
    plt.legend()
    plt.xlabel('Frame number')
    plt.ylabel('Displacement')
    plt.title(f'Displacement over time | {title}')
    plt.grid(True)
    plt.show()


def plot_frequency(frequencies, amplitudes):
    """
    Plots the frequencies and their corresponding amplitudes.

    Args:
        frequencies (numpy.ndarray): Array of frequencies.
        amplitudes (numpy.ndarray): Array of corresponding amplitudes.

    """

    # Only plot for positive frequencies
    mask = frequencies > 0
    freqs = frequencies[mask]
    amps = amplitudes[mask]

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, amps, label='Amplitudes')
    plt.legend()
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('Frequencies after FFT')
    plt.grid(True)
    plt.show()


def plot_hanned_displacements(hanned_displacements):
    """
    Plots the displacements after applying the Hanning window.

    Args:
        hanned_displacements (list): List of displacements after Hanning window.

    """

    t = range(len(hanned_displacements))

    plt.figure(figsize=(10, 6))
    plt.plot(t, hanned_displacements, label='Displacement (Hanning Window)')
    plt.title('Displacements after Hanning Window')
    plt.xlabel('Frame number')
    plt.ylabel('Displacement')
    plt.legend()
    plt.grid(True)
    plt.show()
    

#################################
#
#
#     Tracking centroids methods
#
#
#################################


#basic filtering methods --> works for the sample data 
def get_centroid_basic(frame):
    
    """
    Applies basic filtering methods to find red objects in a frame and returns their centroids.

    Args:
        frame (numpy.ndarray): The input frame in BGR format.

    Returns:
        red_centroids (list): List of tuples representing the centroids of red objects found in the frame.
        marked_frames (list): List of frames with circles drawn at the centroids of red objects.

    """
    
    red_centroids = []
    marked_frames = []
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for red color in HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Bitwise-OR mask1 and mask2
    mask = mask1 | mask2

    # Perform morphological operations to get rid of noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)

        # Find the centroid of the largest contour
        M = cv2.moments(c)
        if M["m00"] != 0:  # check for division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            red_centroids.append((cx, cy))
            #print("Centroid found:", (cx, cy))
            
        # Draw a circle on the frame at the centroid
        marked_frame = cv2.circle(frame.copy(), (cx, cy), 10, (0, 255, 0), 3)
        marked_frames.append(marked_frame)
        #print("Circle drawn at centroid:", (cx, cy))
        
    return red_centroids, marked_frames



#with pixel threshold 
def get_red_centroid_threshold(frame, tolerance, prev_centroid=None):
    """
    Get the centroid of the red color in a frame using thresholding.

    Args:
        frame (numpy.ndarray): Input frame.
        tolerance (float): Tolerance level for creating the HSV bounds.
        prev_centroid (tuple, optional): Previous centroid coordinates. Defaults to None.

    Returns:
        closest_centroid (tuple): Coordinates of the closest centroid.
        marked_frame (numpy.ndarray): Frame with a circle drawn around the closest centroid.

    """

    red_centroids = []
    
    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red color
    red_color = (200, 142, 143)
    red_hsv = rgb_to_hsv(*red_color)
    lower_red, upper_red = create_hsv_bounds(*red_hsv, tolerance=tolerance)

    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a circle on the frame at the centroid
    marked_frame = frame.copy()

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:  # check for division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            red_centroids.append((cx, cy))

    # If there is a previous centroid, find the closest current centroid to it
    if prev_centroid is not None and len(red_centroids) > 0:
        distances = [np.linalg.norm(np.array(centroid) - np.array(prev_centroid)) for centroid in red_centroids]
        closest_centroid = red_centroids[np.argmin(distances)]
    elif len(red_centroids) > 0:  # No previous centroid, find the largest contour
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        closest_centroid = (cx, cy)
    else:
        closest_centroid = []

    # Draw a circle on the frame at the closest centroid
    if len(closest_centroid) > 0:
        marked_frame = cv2.circle(marked_frame, closest_centroid, 15, (0, 255, 0), 3)

    return closest_centroid, marked_frame


#ORB methods 
def get_features(frame):
    """
    Extracts keypoints and descriptors from a given frame using the ORB feature detector.

    Args:
        frame (numpy.ndarray): The input frame in BGR format.

    Returns:
        keypoints (list): List of detected keypoints.
        descriptors (numpy.ndarray): Array of computed descriptors.

    """

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute the descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    """
    Matches descriptors from two different frames using the Brute-Force Matcher (BFMatcher).

    Args:
        descriptors1 (numpy.ndarray): Descriptors from the first frame.
        descriptors2 (numpy.ndarray): Descriptors from the second frame.

    Returns:
        matches (list): List of best matches between the descriptors.

    """

    # Initialize the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors from the first frame and the next frame
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches based on distance. Best matches come first
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def get_features_near_centroid(frame, centroid, max_distance=3):
    """
    Extracts keypoints and descriptors from a given frame, keeping only the ones near a specified centroid.

    Args:
        frame (numpy.ndarray): The input frame in BGR format.
        centroid (list): List of tuples representing the centroid coordinates.
        max_distance (float, optional): The maximum distance allowed from the centroid. Defaults to 3.

    Returns:
        keypoints_near_centroid (list): List of keypoints near the centroid.
        descriptors_near_centroid (numpy.ndarray): Array of descriptors corresponding to the keypoints near the centroid.

    """

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute the descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Keep only the keypoints (and their descriptors) that are within max_distance from the centroid
    keypoints_near_centroid = []
    descriptors_near_centroid = []
    for i, keypoint in enumerate(keypoints):
        # Calculate Euclidean distance between the keypoint and the centroid
        distance = np.sqrt((keypoint.pt[0] - centroid[0][0])**2 + (keypoint.pt[1] - centroid[0][1])**2)
        if distance <= max_distance:
            # Keypoint is within the maximum distance from the centroid, keep it
            keypoints_near_centroid.append(keypoint)
            descriptors_near_centroid.append(descriptors[i])

    return keypoints_near_centroid, np.array(descriptors_near_centroid)


def compute_centroid_of_keypoints(keypoints):
    """
    Computes the centroid of a list of keypoints.

    Args:
        keypoints (list): List of keypoints.

    Returns:
        centroid (tuple): Tuple representing the centroid coordinates (centroid_x, centroid_y).

    """

    # Check if keypoints list is empty
    if not keypoints:
        return None

    # Calculate the total x and y coordinates of keypoints
    total_x = sum(keypoint.pt[0] for keypoint in keypoints)
    total_y = sum(keypoint.pt[1] for keypoint in keypoints)

    # Calculate the centroid coordinates
    centroid_x = total_x / len(keypoints)
    centroid_y = total_y / len(keypoints)

    # Return the centroid as a tuple
    return (centroid_x, centroid_y)




#################################
#
#
#     Displacement data computing 
#             from : https://github.com/MTNakata/AraVib/tree/master/AraVib_modules
#
#
#################################


import numpy as np

def nan_processing(array):
    """
    Process an array with NaN values by filling the NaNs with interpolated values.

    Args:
        array (numpy.ndarray): Input array with NaN values.

    Returns:
        array_copy (numpy.ndarray): Processed array with NaN values filled.

    """

    array_copy = array.copy()
    for i, j in [(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1),
                 (1, 4), (4, 1), (3, 2), (2, 3), (1, 5), (5, 1), (2, 4), (4, 2), (3, 3)]:
        if np.isnan(array_copy).all():
            break
        else:
            array_right = np.roll(array_copy, i, axis=0)
            array_left = np.roll(array_copy, -j, axis=0)
            array_mean = (array_right * j + array_left * i) / (i + j)
            nan_position = np.where(np.isnan(array_copy))
            array_copy[nan_position] = array_mean[nan_position]
    if np.isnan(array_copy[0]).all():
        return array_copy - array_copy[0]
    else:
        return array_copy - np.median(array_copy[:10], axis=0)


def center_to_displacement(center_array):
    """
    Convert an array of center positions to displacement values.

    Args:
        center_array (numpy.ndarray): Array of center positions.

    Returns:
        displacement_array (numpy.ndarray): Array of displacement values.

    """

    displacement_array_0 = np.linalg.norm(center_array, axis=1)
    return np.median(displacement_array_0) - displacement_array_0


def centroids_to_displacements(centroids):
    """
    Convert a list of centroids to an array of displacement values.

    Args:
        centroids (list): List of centroids.

    Returns:
        displacements (numpy.ndarray): Array of displacement values.

    """

    # Convert centroids to numpy array
    centroids_array = np.array(centroids)

    # Process NaNs
    centroids_filled = nan_processing(centroids_array)

    # Calculate displacements
    displacements = center_to_displacement(centroids_filled)

    return displacements

from scipy import fftpack

def displacement_to_difference(displacement_array):
    """
    Compute the difference array from the displacement array.

    Args:
        displacement_array (numpy.ndarray): Array of displacement values.

    Returns:
        dif_array (numpy.ndarray): Array of differences between consecutive displacement values.
        start_point (int): Start index of the maximum difference.

    """

    displacement_array_copy = displacement_array.copy()
    dif_array = displacement_array_copy - np.roll(displacement_array_copy, -1)
    dif_array = dif_array[:-1]
    start_point = np.min(np.where(dif_array == np.max(dif_array)))
    return dif_array, start_point


def transform_hanning(displacement_array, start_point):
    """
    Apply the Hanning window to the displacement array.

    Args:
        displacement_array (numpy.ndarray): Array of displacement values.
        start_point (int): Start index for the Hanning window.

    Returns:
        displacement_array_hanning (numpy.ndarray): Array of displacement values after applying the Hanning window.

    """

    hanning_length = len(displacement_array) - 2 * start_point
    hanningWindow = np.concatenate([np.zeros(start_point), np.hanning(hanning_length), np.zeros(start_point)])
    displacement_array2 = displacement_array - np.mean(displacement_array[start_point:])
    displacement_array_hanning = displacement_array2 * hanningWindow
    return displacement_array_hanning


def displacement_to_major_freq(displacement_array, fps=240):
    """
    Compute the major frequency from the displacement array using FFT.

    Args:
        displacement_array (numpy.ndarray): Array of displacement values.
        fps (int, optional): Frames per second. Defaults to 240.

    Returns:
        freqs (numpy.ndarray): Array of frequencies.
        power (numpy.ndarray): Array of corresponding power values.
        major_freq (float): Major frequency from the FFT.

    """

    time_step = 1 / fps
    freq = fftpack.fftfreq(displacement_array.size, d=time_step)
    fft = fftpack.fft(displacement_array)
    pidxs = np.where(freq > 0)
    freqs = freq[pidxs]
    power = np.abs(fft)[pidxs]
    freqs2 = freqs[np.where(freqs > 2)]
    power2 = power[np.where(freqs > 2)]
    major_freq = freqs2[np.where(power2 >= max(power2))]
    return freqs, power, major_freq


#update hanning function to adapat it to any array 

def transform_hanning_adapt(displacement_array, start_point):
    """
    Apply the Hanning window to the displacement array with adaptive padding.

    Args:
        displacement_array (numpy.ndarray): Array of displacement values.
        start_point (int): Start index for the Hanning window.

    Returns:
        displacement_array_hanning (numpy.ndarray): Array of displacement values after applying the Hanning window.

    """

    # Get the length of the displacement array
    array_length = len(displacement_array)

    # Subtract the mean of the displacement array from the displacement array itself
    displacement_array2 = displacement_array - np.mean(displacement_array[start_point:])

    # If the displacement array is shorter than 240, return the mean-subtracted array
    if array_length < 240:
        return displacement_array2

    # Length of the Hanning window
    hanning_length = min(240, array_length)

    # The length of the zero-padding on either side of the hanning window
    pad_length = (array_length - hanning_length + 1) // 2  # added 1 to ensure correct padding

    # In case of odd total length, one padding needs to be longer
    if (pad_length * 2 + hanning_length) != array_length:
        pad_length_1 = pad_length
        pad_length_2 = pad_length + 1
    else:
        pad_length_1 = pad_length_2 = pad_length

    # Create the Hanning window with zero-padding
    hanningWindow = np.concatenate([np.zeros(pad_length_1), np.hanning(hanning_length), np.zeros(pad_length_2)])

    # Multiply the displacement array by the Hanning window
    displacement_array_hanning = displacement_array2 * hanningWindow
    return displacement_array_hanning



