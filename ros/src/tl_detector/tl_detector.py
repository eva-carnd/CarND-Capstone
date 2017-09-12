#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import math
import yaml
import numpy as np
from traffic_light_config import config

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.num_waypoints = 0
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        #sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and 
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.test_image_pub = rospy.Publisher('/test_image', Image, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints): # type: Lane
        self.waypoints = waypoints
        self.num_waypoints = len(self.waypoints.waypoints)
        self.base_waypoints_sub.unregister()

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        rospy.loginfo("""Lights wp: {}  state: {}""".format(light_wp, state))


        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        waypoints = self.waypoints.waypoints

        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)#  + (a.z-b.z)**2)

        closest_len = 10000000
        
        
        closest_waypoint = 0 
        next_waypoint = 0
            
        num_waypoints = len(waypoints)
        dist = dl(waypoints[closest_waypoint].pose.pose.position, pose.position)

        while (dist < closest_len) and (closest_waypoint < num_waypoints):
            closest_waypoint = next_waypoint
            closest_len = dist
            dist = dl(waypoints[closest_waypoint+1].pose.pose.position, pose.position)
            next_waypoint += 1

        dist_prev = dl(waypoints[closest_waypoint-1].pose.pose.position, pose.position)
        dist_curr = dl(waypoints[closest_waypoint].pose.pose.position, pose.position)
        dist_next = dl(waypoints[closest_waypoint+1].pose.pose.position, pose.position)

        # rospy.loginfo("""Light detection -> {} waypoint dist {} {} {}""".format(closest_waypoint, dist_prev, dist_curr, dist_next))

        return closest_waypoint


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image
        # for debugging purposes - use 'rqt_image_view' in a separate terminal to see the image
        img = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
        self.test_image_pub.publish(img)

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def euclidean_distance(self, p1x, p1y, p2x, p2y):
        x_dist = p1x - p2x
        y_dist = p1y - p2y
        return math.sqrt(x_dist*x_dist + y_dist*y_dist)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_positions = self.config['light_positions']
        light_wp = 0

        if(self.pose and self.waypoints):
            closest_waypoint_index = self.get_closest_waypoint(self.pose.pose)
            closest_waypoint_ps = self.waypoints.waypoints[closest_waypoint_index].pose

            #TODO find the closest visible traffic light (if one exists)
            closest_light_position = None
            closest_light_distance = float("inf")
            for light_position in config.light_positions:
                distance = self.euclidean_distance(light_position[0], light_position[1], closest_waypoint_ps.pose.position.x, closest_waypoint_ps.pose.position.y)
                if distance < closest_light_distance:
                    closest_light_distance = distance
                    closest_light_position = light_position

            # if the traffic light is within a distance of '50' then process the pose
            if closest_light_distance < 80:
                light_pose = Pose()
                light_pose.position.x = closest_light_position[0]
                light_pose.position.y = closest_light_position[1]
                light_pose.position.z = 0
                light_wp = self.get_closest_waypoint(light_pose)

                # only keep the traffic light if it is in front of the car
                if light_wp >= closest_waypoint_index:
                    light = light_pose

                rospy.loginfo("""closest -> {} : {} {} {} {}""".format(closest_light_distance, closest_light_position[0], closest_light_position[1], closest_waypoint_ps.pose.position.x, closest_waypoint_ps.pose.position.y))
            else:
                light = None


        if light:
            state = self.get_light_state(light)
            return light_wp, state
        # self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
