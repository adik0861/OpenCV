import cv2 as cv2
import imutils


def gauss_frame(frame_in, kernel_size):
    gray = cv2.cvtColor(frame_in, cv2.COLOR_BGR2GRAY)
    kernel_size = (kernel_size, kernel_size)  # (kernel width, kernel height) should be odd
    std_dev = 0
    gray = cv2.GaussianBlur(gray, kernel_size, std_dev)
    return gray


def get_threshold(current_frame, previous_frame, kernel_size):
    prev_frame_grey = gauss_frame(previous_frame, kernel_size)
    current_frame_grey = gauss_frame(current_frame, kernel_size)
    abs_diff = cv2.absdiff(prev_frame_grey, current_frame_grey)
    thresh = cv2.threshold(abs_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    return thresh


def get_bbox(thresh, frame_in, frame_out=None, min_bbox_size=100):
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if frame_out is None:
        frame_out = frame_in
    frame_out = cv2.cvtColor(frame_out, cv2.COLOR_GRAY2BGR)
    for c in cnts:
        if cv2.contourArea(c) < min_bbox_size:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame_out


camera = cv2.VideoCapture(0)
_, old_frame = camera.read()
while True:
    _, frame = camera.read()
    contour_diff = get_threshold(current_frame=frame, previous_frame=old_frame, kernel_size=3)
    pretty_diff = get_threshold(current_frame=frame, previous_frame=old_frame, kernel_size=31)
    bboxed_image = get_bbox(
        thresh=contour_diff,
        frame_in=contour_diff,
        frame_out=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        min_bbox_size=4000)
    cv2.imshow('my webcam', bboxed_image)
    old_frame = frame
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
