import imutils
import numpy as np
import cv2
import matplotlib.pyplot as plt


def capture_frame():
    camera = cv2.VideoCapture(0)  # use 0 for web camera.py
    success, frame = camera.read()
    return frame


def gauss_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return gray


def display_frame(frame, transform=False):
    if transform:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame)
    plt.axis('off')
    plt.show()


def save_frame(frame: np.ndarray, file_path: str = 'data/frame'):
    with open(file_path, 'wb') as f:
        np.save(f, frame)
        print('Frame Saved!')


def load_frame(file_name='frame'):
    with open(file_name, 'rb') as f:
        _frame = np.load(f)
        print('Frame Loaded!')
        return _frame


def get_threshold(current_frame, previous_frame):
    prev_frame_grey = gauss_frame(previous_frame)
    current_frame_grey = gauss_frame(current_frame)
    abs_diff = cv2.absdiff(prev_frame_grey, current_frame_grey)
    thresh = cv2.threshold(abs_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    return thresh


if __name__ == '__main__':
    from time import time

    f1 = capture_frame()
    f2 = capture_frame()
    display_frame(f1, transform=True)
    display_frame(f2, transform=True)

    g1 = gauss_frame(f1)
    g2 = gauss_frame(f2)
    abs_diff = cv2.absdiff(g1, g2)
    display_frame(abs_diff)
    thresh = cv2.threshold(abs_diff, 25, 255, cv2.THRESH_BINARY)[1]
    display_frame(thresh)
    thresh = cv2.dilate(thresh, None, iterations=2)
    display_frame(thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        print(cv2.contourArea(c))
        # if cv2.contourArea(c) < args["min_area"]:
        #     continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(f2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"
        cv2.imshow("Security Feed", f2)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", abs_diff)
