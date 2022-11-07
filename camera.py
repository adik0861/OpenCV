import cv2 as cv2
import imutils


def gauss_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel_size = (7, 7)  # (kernel width, kernel height) should be odd
    std_dev = 0
    gray = cv2.GaussianBlur(gray, kernel_size, std_dev)
    return gray


def get_threshold(current_frame, previous_frame):
    prev_frame_grey = gauss_frame(previous_frame)
    current_frame_grey = gauss_frame(current_frame)
    abs_diff = cv2.absdiff(prev_frame_grey, current_frame_grey)
    thresh = cv2.threshold(abs_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    return thresh


def get_bbox(frame_in, thresh, min_bbox_size=100):
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    frame_out = cv2.cvtColor(frame_in, cv2.COLOR_GRAY2BGR)
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
    diff = get_threshold(current_frame=frame, previous_frame=old_frame)
    bboxed_image = get_bbox(frame_in=diff, thresh=diff, min_bbox_size=1000)
    # diff.image(bboxed_image, channels="BGR")  # OpenCV images are BGR!
    cv2.imshow('my webcam', bboxed_image)
    old_frame = frame
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
