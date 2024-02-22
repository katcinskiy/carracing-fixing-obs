import os

import cv2
import numpy as np


def create_video_from_frames(frames, output_video_path, fps=50):
    # Check if the frame list is empty
    if not frames:
        print("The frame list is empty. No video will be created.")
        return
    if os.path.dirname(output_video_path):  # Check if the directory path is not empty
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    # Determine the size of the first frame
    height, width = frames[0].shape[:2]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 file
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert frames to the correct color format if necessary
        frame = frame.astype(np.uint8)
        if frame.ndim == 2 or frame.shape[2] == 1:
            # If the frame is grayscale, convert it to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            # If the frame has an alpha channel, convert it to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Ensure the frame size matches the video size
        if (frame.shape[0] != height) or (frame.shape[1] != width):
            print("Frame size does not match video size. Resizing frame.")
            frame = cv2.resize(frame, (width, height))

        out.write(frame)

    out.release()


def draw_trajectory_on_frame(frame, car_states):

    if len(car_states) < 50:
        return

    height, width = frame.shape[:2]
    half_width = int(width / 2)
    quarter_height = int(height / 4)


    new_positions = [(0., 0.)]

    for item in reversed(car_states[:-1]):
        new_positions.append((car_states[-1]['x'] - item['x'], car_states[-1]['y'] - item['y']))

    cur_i = 1
    while np.linalg.norm([new_positions[cur_i][0], new_positions[cur_i][1]]) < 0.01:
        cur_i += 1

    angle = np.arctan2(new_positions[cur_i][0], new_positions[cur_i][1])

    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    new_positions = [np.dot(rotation_matrix, np.array(vector)) for vector in new_positions]


    pointsInside = np.array([
        (np.clip(half_width - 4 * item[0], 0, 400).astype(np.int32), np.clip(height - quarter_height + 4 * item[1], 0, 400).astype(np.int32))
        for item in new_positions])

    for index, item in enumerate(pointsInside):
        if index == len(pointsInside) - 1:
            break
        cv2.line(frame, item, pointsInside[index + 1], [255, 255, 255], 1)



def draw_full_trajectory(car_states):
    new_positions = [(0., 0.)]

    for item in reversed(car_states[:-1]):
        new_positions.append((car_states[-1]['x'] - item['x'], car_states[-1]['y'] - item['y']))

    cur_i = 1
    while np.linalg.norm([new_positions[cur_i][0], new_positions[cur_i][1]]) < 0.01:
        cur_i += 1

    angle = -np.arctan2(new_positions[cur_i][0], new_positions[cur_i][1])

    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])

    plt.plot(*zip(*new_positions))
    new_positions = [np.dot(rotation_matrix, np.array(vector)) for vector in new_positions]
    plt.plot(*zip(*new_positions))
    plt.show()
    print(f"Angle = {angle}")