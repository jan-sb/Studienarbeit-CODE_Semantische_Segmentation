import cv2 as cv
import numpy as np
import sys
import datetime
import os




def watershed_segmentation(image):
    assert image is not None, "file could not be read, error in watershed segmentation"
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(image, markers)
    segmented = np.zeros_like(image)
    for i in range(2, markers.max() + 1):
        segmented[markers == i] = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]

    segmented = cv.addWeighted(image, 1, segmented, 0.3, gamma=0)

    image[markers == -1] = [255, 0, 0]
    markers = np.uint8(markers)

    return segmented, markers



def cap_def(path):
    cap = cv.VideoCapture(path)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv.CAP_PROP_FPS))

    return cap, width, height, length, fps




def update_console(message):
    os.system('cls' if os.name == 'nt' else 'clear')
    sys.stdout.write(f"\r{message}")
    sys.stdout.flush()



def video_writer(output_path, fps, resolution_tuple):
    video_title = f'video_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    output_path_final = os.path.join(output_path, video_title)
    out = cv.VideoWriter(output_path_final, fourcc, fps, resolution_tuple)
    print(f'Video writer initialized: {output_path}/{video_title}, with {fps} fps and resolution {resolution_tuple}, fourcc: {fourcc}')
    return out

def update_progress_bar(current_frame, max_frames, bar_length=20):
    ratio = current_frame / max_frames
    progress = int(bar_length * ratio)
    bar = "[" + "=" * progress + " " * (bar_length - progress) + "]"
    percentage = int(ratio * 100)
    return f"{bar} {percentage}%"