import cv2 as cv2


def gauss_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel_size = (11, 11)  # (kernel width, kernel height) should be odd
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


camera = cv2.VideoCapture(0)
_, old_frame = camera.read()
while True:
    _, frame = camera.read()
    diff = get_threshold(current_frame=frame, previous_frame=old_frame)
    cv2.imshow('my webcam', diff)
    old_frame = frame
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
