# coding: utf-8
'''
	The Track class as well as TrackPosition which contains important information about location of car in track is implemented here
'''
import random
import numpy as np
import math
import rospy
import sys

from tf.transformations import euler_from_quaternion
from track_geometry import PATH

from collections import namedtuple
'''
Returns a new tuple subclass named Point. The new subclass is used to create tuple-like objects that have fields accessible by attribute lookup as well as being indexable and iterable
'''
Point = namedtuple("Point", ["x", "y"])

class TrackPosition():
    def __init__(self, segment, x, y, point, track):
        self.point = point
        self.segment = segment
        self.distance_to_center = x 
        self.segment_distance = y # distance moved in cuurent segment
        self.segment_progress = y / track.segment_length[segment] # progress done in current segment
        self.total_distance = track.cumulative_distance[segment] + y 
        self.total_progress = self.total_distance / track.length
        self.angle = math.atan2(
            track.forward[segment, 1], track.forward[segment, 0]) # angle of the segment to horizontal, not completely understood

    def __str__(self): # This is what will be returned when you call the print() function on an object of type TrackPosition
        return "{0:.2f} m ({1:.0f}%), segment {2:d}, to center: {3:.2f} m, track: {4:.0f}°".format(
            self.total_distance,
            self.total_progress * 100,
            self.segment,
            self.distance_to_center,
            self.angle * 180 / math.pi
        )
		
    def get_relative_angle(self, orientation):
        quaternion = [
            orientation.w,
            orientation.x,
            orientation.y,
            orientation.z]
        euler = euler_from_quaternion(quaternion) # orientation with world frame, (x to left and y upward)
        return (euler[0] + self.angle) % (2 * math.pi) - math.pi 

    def faces_forward(self, orientation):
        return abs(self.get_relative_angle(orientation)) < math.pi / 2 # magniutde less than 90deg, then we face forward


class Track():
    def __init__(self, points):
        self.points = points[:-1, :] # just copying of the points to the attribute except last point
        self.size = points.shape[0] - 1	# size is no. of points -1 since this is the resulting number of vectors 
        relative = points[1:, :] - points[:-1, :] # the difference between each point and the consecutive point stored as an array
        self.segment_length = np.linalg.norm(relative, axis=1) # magnitude(len) each vector, since axis=1 so each row is considered a vector
        self.length = np.sum(self.segment_length) # total length of the vectors
        self.forward = relative / self.segment_length[:, np.newaxis] # getting the normalized vectorized, just direction info is preserved
        self.right = np.array(
            [self.forward[:, 1], -self.forward[:, 0]]).transpose() # rotation of forward right by 90 degrees
        self.cumulative_distance = np.zeros(points.shape[0])
        self.cumulative_distance[1:] = np.cumsum(self.segment_length) # Cumlative sum of segments
		# NEED TO UNDERSTAND HOW BOTTOM TWO METHODS WORK #
    def localize(self, point):
        ''' Returns a TrackPosition object based on a position in world space. '''
        local = np.array([point.x, point.y]) - self.points
        x = local[:, 0] * self.right[:, 0] + local[:, 1] * self.right[:, 1]
        y = local[:, 0] * self.forward[:, 0] + local[:, 1] * self.forward[:, 1]
        distances = np.abs(x)
        distances[(y < 0) | (y > self.segment_length)] = float("Inf")
        segment = np.argmin(distances)
        return TrackPosition(segment, x[segment], y[segment], point, self)

    def get_position(self, total_distance, distance_to_center=0):
        ''' Returns a TrackPosition object based on the on-track distance from the start
        line and the distance to the center of the track. '''
        segment = 0
        while self.cumulative_distance[segment + 1] < total_distance:
            segment += 1
        segment_distance = total_distance - self.cumulative_distance[segment]
        x = self.points[segment, 0] + self.forward[segment, 0] * \
            segment_distance + self.right[segment, 0] * distance_to_center
        y = self.points[segment, 1] + self.forward[segment, 1] * \
            segment_distance + self.right[segment, 1] * distance_to_center
        return TrackPosition(segment, distance_to_center,  # nopep8
            segment_distance, Point(x, y), self)  # nopep8


world_name = rospy.get_param("world_name")

if world_name not in [
    "racetrack_decorated",
    "racetrack_decorated_2",
        "racetrack_decorated_2_big"]:
    print "ERROR: Racetrack not supported by track.py"
    sys.exit(1)

if world_name == "racetrack_decorated_2_big":
    PATH *= 2.5

track = Track(PATH)
