#! /usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os

from time import time
from math import cos, sin, pi, atan2, sqrt, floor
from typing import Tuple


def make_video(frames: int, duration: float):
    """ Make video from folder containing images.
        :param frames: number of frames in the video
        :param duration: duration of video in seconds"""

    image_folder = '/home/jetson/catkin_ws/src/jetracer_ros/src/video'
    video_name = 'lidar_view.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv.VideoWriter(video_name, 0, round(frames/duration), (width, height))

    for image in images:
        video.write(cv.imread(os.path.join(image_folder, image)))

    cv.destroyAllWindows()
    video.release()
    print("Video filepath : " + image_folder + "/" + video_name)


class lidarMap:
    def __init__(self, lrange: float):
        """ Making x, y map of obstacles, route production.
            :param lrange: range of lidar in meters"""

        self.obstacles = []
        self.car_loc = [0, 0]
        self.route = [0, 0]
        self.prev_maps = []  # previous maps data
        self.lrange = lrange  # lidar range

    def clear(self):
        """ Clearing the data. """
        self.obstacles = []
        self.route = []

    def feed(self, x: int, y: int):
        """ Passing the obstacle point to map.
            :param x: x location of obstacle in the map
            :param y: y location of obstacle in the map"""

        self.obstacles.append((x, y))

    def save(self, frames: int, duration: float):
        """ Saving the images, making the video.
            :param frames: number of frames
            :param duration: duration of run in seconds"""

        print("Saving scans")
        save_length = len(self.prev_maps)
        for ind, ma in enumerate(self.prev_maps):
            plt.plot(self.car_loc[0], self.car_loc[1], 'ro', markersize=4)
            plt.plot([self.car_loc[0], self.car_loc[0]], [self.car_loc[0], self.car_loc[1] + 0.5], 'r-', linewidth=2)

            try:
                if not ma[2]:
                    ma[2] = len(ma[1]) / 2
            except IndexError:
                ma.append(len(ma[1]) / 2)

            for indi, rou in enumerate(ma[1]):
                if not indi == ma[2]:
                    plt.plot(rou[0], rou[1], 'go', markersize=1)
                else:
                    plt.plot(rou[0], rou[1], 'go', markersize=2, color="m")

            for x, y in ma[0]:
                plt.plot(x, y, 'ko', markersize=0.5)

            plt.xlim([-0.1, self.lrange / 100])
            plt.ylim([-0.1, self.lrange / 100])
            filename = "/home/jetson/catkin_ws/src/jetracer_ros/src/video/prev_map" + str(ind) + ".png"
            plt.savefig(filename)
            plt.clf()

            print(str(ind / save_length) + "% of scans saved")

        print("Making video")
        make_video(frames, duration)
        print("Made video")

    def add_lookahead(self, lookahead: int):
        """ Informing to which point is the car heading.
            :param lookahead: index in route of desired point"""

        try:
            self.prev_maps[len(self.prev_maps) - 1].append(lookahead)
        except IndexError as err:
            print(err)

    def add_prev(self):
        """ Saving current map data. """

        self.prev_maps.append([self.obstacles, self.route])

    def make_route(self):
        """ Calculating the optimal route. """

        xs = []
        ys = []
        for x, y in self.obstacles:  # convert meters into centimeters
            xs.append(int(round(x * 100)))
            ys.append(int(round(y * 100)))
        data_circles = np.zeros((self.lrange, self.lrange))  # create the image for map

        # lidar error
        mx_error = 20  # compensate lidar error
        for ind, val in enumerate(xs):
            cv.circle(data_circles, (val, ys[ind]), mx_error, 255, -1)

        img = np.array(data_circles, np.uint8)  # Convert random_data to NumPy array of type 'uint8'
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv.dilate(img, kernel, iterations=5)  # make the obstacles larger
        edges = cv.bitwise_not(img_dilation)  # negation

        cnts = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # split image into areas
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        correct_contour = 0

        # Perform check if point is inside contour/shape
        for ind, c in enumerate(cnts):
            if cv.pointPolygonTest(c, (2, 2), False) == 1:  # determine the contour in which the car is
                correct_contour = ind

        route = []
        for ind, point in enumerate(cnts[correct_contour]):
            if (ind % 5) == 0:  # append every fifth point to the route
                if not (point[0][0] == 0) and not (point[0][1] == 0) \
                        and not (point[0][0] == self.lrange - 1) and not (point[0][1] == self.lrange - 1):
                    route.append([point[0][0], point[0][1]])

        for ind, route_point in enumerate(route):
            # converting and subtracting, thanks to subtracting,the route will always be visible to the car
            route[ind] = [float(route_point[0]) / 100 - 0.2, float(route_point[1]) / 100 - 0.2]

        route.reverse()
        self.route = route


class purePursuit:
    def __init__(self, lookahead_distance: float, robot_length: float):
        """ Route following algorithm.
            :param lookahead_distance: meters, coefficient\n
            :param robot_length: meters, constant """

        self.lookahead_distance = lookahead_distance
        self.robot_length = robot_length

    def calc(self, robot_pose: list, path: list) -> Tuple[float, int]:
        """ Calculating necessary angle to follow the route.
            :param robot_pose: (x, y, theta)
            :param path: route
            :returns: steer_angle, desired point index"""

        closest_point_index = int(floor(self.find_closest_point(robot_pose, path)))
        lookahead_point_index = int(floor(self.find_lookahead_point(robot_pose, path, closest_point_index)))

        lookahead_point = path[lookahead_point_index]

        alpha = atan2(lookahead_point[1] - robot_pose[1], lookahead_point[0] - robot_pose[0]) - robot_pose[2]
        steer_angle = atan2(2.0 * self.robot_length * sin(alpha) / self.lookahead_distance, 1.0)
        return steer_angle, lookahead_point_index

    def find_closest_point(self, robot_pose: list, path: list) -> int:
        """ Finding the closest point in the route.
            :param robot_pose: (x, y, theta)
            :param path: route
            :returns: closest point index"""

        closest_distance = float('inf')
        closest_index = 0

        for i, point in enumerate(path):
            distance = sqrt((point[0] - robot_pose[0]) ** 2 + (point[1] - robot_pose[1]) ** 2)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i

        return closest_index

    def find_lookahead_point(self, robot_pose: list, path: list, closest_point_index: int) -> int:
        """ Finding point in desired distance in the route.
            :param robot_pose: (x, y, theta)
            :param path: route
            :param closest_point_index: index in path
            :returns: desired point index"""

        path_length = len(path)
        lookahead_distance_sq = self.lookahead_distance ** 2

        for i in range(closest_point_index, path_length):
            dx = path[i][0] - robot_pose[0]
            dy = path[i][1] - robot_pose[1]
            distance_sq = dx ** 2 + dy ** 2

            if distance_sq > lookahead_distance_sq:
                return i

        return path_length - 1  # Return last point if no point is farther than lookahead distance


class car:
    def __init__(self, algorithm: purePursuit, area: lidarMap, speed: float, turn: float, lrange: float):
        """ Class for communication with the car.
            :param algorithm: algorithm used for route following
            :param area: mapping class
            :param speed: initial speed
            :param turn: initial turn
            :param lrange: lidar range"""

        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)  # publisher, a "command link" to real car
        self.speed = speed  # desired speed, later published to car
        self.turn = turn  # desired turn angle of front axis, later published to car

        self.distances = []  # lidar scan results
        self.lrange = lrange  # the maximum distance to which the measurements will be processed

        self.map = area  # mapping class, for lidar scan conversion

        self.algorithm = algorithm  # route following algorithm

        # time measurements
        self.start = int(time() * 1000.0)
        self.since_start = 0

        # video generation
        self.lookahead = 0  # point which the algorithm is following
        self.loop_num = 0  # iteration number

    def go(self):
        """ Start function """

        print("JetRacer Ready")
        # node setting
        rospy.init_node('scan_values')
        rospy.Subscriber('/scan', LaserScan, self.detect_and_react)
        rospy.spin()

    def detect_and_react(self, msg):
        """ Main loop, getting the measurements, mapping, moving."""

        self.get_dist(msg)  # getting measurements from lidar

        # calculating turn necessary to follow route
        self.turn, self.lookahead = self.algorithm.calc([0, 0, pi / 2], self.map.route)
        self.move()  # publish turn and speed to the car

        self.since_start = int(time() * 1000.0) - self.start  # time since start

    def get_dist(self, msg):
        """ Getting the lidar measurements, conversion to cartesian and feeding the mapping class."""

        one_deg = 1146.0 / 360.0
        view_ang = 270.0  # cutting the angle range of lidar

        self.map.clear()  # clearing the map for next measurements

        for ind, meas in enumerate(msg.ranges):
            angle = ind * (1 / one_deg)  # calculating the angle
            length = meas  # getting the distance from detected obstacle

            # converting from polar coordinates to cartesian
            alfa = (angle - view_ang) * pi / 180
            x = length * cos(alfa)
            y = length * sin(alfa)

            # limiting the achieved measurements
            if (x < self.lrange / 100) and (x > -self.lrange / 100):
                if (y < self.lrange / 100) and (y > -self.lrange / 100):
                    if (x > 0) and (y > 0):
                        self.map.feed(x, y)

        self.map.make_route()  # calculating route
        self.map.add_lookahead(self.lookahead)  # adding lookahead point to make the image
        self.map.add_prev()  # saving finished map data

        if self.loop_num > 270:  # limiting the iterations number and thus the program time duration
            self.map.save(self.loop_num, self.since_start / 1000)  # Saving maps from this run
            print("Saving finished")
            exit()

        self.loop_num = self.loop_num + 1  # incrementing the iterations number

    def move(self):
        """ Passing parameters to the real car. """

        twist = Twist()
        twist.linear.x = self.speed
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = self.turn
        self.pub.publish(twist)  # sending desired parameters to the car


try:
    # initialize necessary classes, pass the parameters
    regulation = purePursuit(lookahead_distance=0.6, robot_length=0.15)
    area_scan = lidarMap(lrange=100)
    jet = car(algorithm=regulation, area=area_scan, speed=0.8, turn=0, lrange=100)

    jet.go()  # starting the run
except KeyboardInterrupt:
    print("Program interrupted by user")

