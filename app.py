from flask import Flask, render_template, Response
# noinspection PyUnresolvedReferences
import numpy as np
import cv2 as cv2
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
from imutils.video import FileVideoStream
from data.frame_io import get_threshold

# from imutils.video import FPS


app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera.py


# camera.py = FileVideoStream(0, queue_size=12).start()

def show_camera():
    camera = cv2.VideoCapture(0)  # use 0 for web camera.py
    _, old_frame = camera.read()  # read the camera.py frame
    while True:

        # Capture frame-by-frame
        # if isinstance(camera.py, FileVideoStream):
        #     frame = camera.py.read()  # read the camera.py frame

        _, frame = camera.read()  # read the camera.py frame
        diff = get_threshold(current_frame=frame, previous_frame=old_frame)
        cv2.imshow('my webcam', diff)
        old_frame = frame
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


def gen_frames():  # generate frame by frame from camera.py
    _, old_frame = camera.read()  # read the camera.py frame
    while True:
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:  # press 'ESC' to quit
        #     break

        # Capture frame-by-frame
        # if isinstance(camera.py, FileVideoStream):
        #     frame = camera.py.read()  # read the camera.py frame

        _, frame = camera.read()  # read the camera.py frame
        diff = get_threshold(current_frame=frame, previous_frame=old_frame)
        cv2.imshow('my webcam', diff)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        old_frame = frame
    cv2.destroyAllWindows()

    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



cv2.destroyAllWindows()
camera.stop()


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=8885)
