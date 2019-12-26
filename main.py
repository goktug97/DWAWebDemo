import time

import cv2
import dwa
import numpy as np
from flask import Flask, Response, render_template, request
from flask import stream_with_context


class WebDemo(object):
    def __init__(self):
        self.app = Flask(__name__)
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/frame', 'frame', self.frame)
        self.app.add_url_rule('/mouse_events', 'mouse', self.mouse, methods=['POST'])
        self.app.add_url_rule('/key_events', 'keypress', self.keypress, methods=['POST'])

        self.drawing = False
        self.draw_points = []
        self.point_cloud = []

        self.vel = (0.0, 0.0)
        self.pose = (30.0, 30.0, 0)
        self.goal = None
        self.base = [3.0, 2.5, -3.0, -2.5]
        self.config = dwa.Config(
                max_speed = 3.0,
                min_speed = -1.0,
                max_yawrate = np.radians(40.0),
                max_accel = 15.0,
                max_dyawrate = np.radians(110.0),
                velocity_resolution = 0.1,
                yawrate_resolution = np.radians(1.0),
                dt = 0.1,
                predict_time = 3.0,
                heading = 0.15,
                clearance = 1.0,
                velocity = 1.0,
                base = self.base)

    def index(self):
        return render_template('index.html')

    def keypress(self):
        data = request.get_json(force=True)
        key = data['key']
        if key == 'r':
            self.draw_points = []
            self.point_cloud = []
            self.vel = (0.0, 0.0)
            self.pose = (30.0, 30.0, 0)
        return ''

    def frame(self):
        @stream_with_context
        def generate():
            while True:
                frame = np.ones((600, 600, 3), dtype=np.uint8)
                time.sleep(0.01)
                for point in self.draw_points:
                    cv2.circle(frame, tuple(point), 4, (255, 255, 255), -1)
                if self.goal is not None:
                    cv2.circle(frame, (int(self.goal[0]*10), int(self.goal[1]*10)),
                            4, (0, 255, 0), -1)
                    if len(self.point_cloud):
                        # Planning
                        self.vel = dwa.planning(self.pose, self.vel, self.goal,
                                np.array(self.point_cloud, np.float32), self.config)
                        # Simulate motion
                        self.pose = dwa.motion(self.pose, self.vel, self.config.dt)

                pose = np.ndarray((3,))
                pose[0:2] = np.array(self.pose[0:2]) * 10
                pose[2] = self.pose[2]

                base = np.array(self.base) * 10
                base[0:2] += pose[0:2]
                base[2:4] += pose[0:2]

                # Not the correct rectangle but good enough for the demo
                width = base[2] - base[0]
                height = base[3] - base[1]
                rect = ((pose[0], pose[1]), (width, height), np.degrees(pose[2]))
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (0,0,255), -1)

                enc_frame = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'\r\n' + b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + enc_frame + b'\r\n')
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def mouse(self):
        data = request.get_json(force=True)
        x = data['x']
        y = data['y']
        click = data['click']
        if click:
            self.drawing = True
        elif click == False:
            self.drawing = False
        elif click is None:
            if self.drawing:
                if [x, y] not in self.draw_points:
                    self.draw_points.append([x, y])
                    self.point_cloud.append([x/10, y/10])
                    self.goal = None
            else:
                self.goal = (x/10, y/10)
        return ""

    @property
    def application(self):
        return self.app

app = WebDemo().application

if __name__ == '__main__':
    app.run()
