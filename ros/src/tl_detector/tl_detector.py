#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf # TODO might be obsolete
import tf2_ros
import tf2_geometry_msgs
import cv2
import math
from traffic_light_config import config

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights helps you acquire an accurate ground truth data source for the traffic light
        classifier, providing the location and current color state of all traffic lights in the
        simulator. This state can be used to generate classified images or subbed into your solution to
        help you work on another single component of the node. This topic won't be available when
        testing your solution in real life so don't rely on it in the final submission.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/camera/image_raw', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
	self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints): # type: Lane
        self.waypoints = waypoints

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
        return 0


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

	pose_in_world = PoseStamped()
	pose_in_world.pose.position.x = point_in_world[0]
	pose_in_world.pose.position.y = point_in_world[1]
	pose_in_world.pose.position.z = 2.0 #point_in_world[2] TODO info is missing in sim!

        fx = config.camera_info.focal_length_x
        fy = config.camera_info.focal_length_y

        image_width = config.camera_info.image_width
        image_height = config.camera_info.image_height

        # get transform between pose of camera and world frame
        trans = None
        try:
            target_frame = "base_link"
            source_frame = "world" 
            transform = self.tf_buffer.lookup_transform(target_frame,
                                       source_frame,
                                       rospy.Time(0), #get the tf at first available time
                                       rospy.Duration(1.0)) #wait for 1 second

            pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_in_world, transform) # TODO point might not actually be a pose

        except (tf.Exception, tf.LookupException, tf.ConnectivityException): # TODO tf2 will throw different exceptions
            rospy.logerr("Failed to find camera to map transform")
        
        camera_x = pose_transformed.pose.position.x
        camera_y = pose_transformed.pose.position.y
        camera_z = pose_transformed.pose.position.z

        # Use tranform and rotation to calculate 2D position of light in image
        # From: https://www.scratchapixel.com/lessons/3d-basic-rendering/computing-pixel-coordinates-of-3d-point/mathematics-computing-2d-coordinates-of-3d-points
        
        screen_x = camera_x / -camera_z
        screen_y = camera_y / -camera_z
        
        # check if point is actually visible
        # assuming fx is canvasWidth, fy is canvasHeight
        if (abs(screen_x) > fx or abs(screen_y) > fy): 
            # return false;
            # TODO figure out how to deal with the tl not being visible
            pass
        
        # normalize to [0,1]
        norm_screen_x = (screen_x + fx / 2) / fx
        norm_screen_y = (screen_y + fy / 2) / fy
        
        # convert to pixel coordinates
        pix_x = math.floor(norm_screen_x * image_width)
        pix_y = math.floor((1-norm_screen_y) * image_height)
        
        return (pix_x, pix_y)

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

        x, y = self.project_to_image_plane(light)

	print("Traffic light position in image:",x, y)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def euclidean_distance(self, p1x, p1y, p2x, p2y):
        x_dist = p1x - p2y
        y_dist = p1x - p2y
        return math.sqrt(x_dist*x_dist + y_dist*y_dist)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light_position = None
        light_positions = config.light_positions
        if(self.pose and self.waypoints):
            closest_waypoint_index = self.get_closest_waypoint(self.pose.pose)
            closest_waypoint_ps = self.waypoints.waypoints[closest_waypoint_index].pose

            #TODO find the closest visible traffic light (if one exists)
            closest_light_distance = float("inf")
            for light_position in config.light_positions:
                distance = self.euclidean_distance(light_position[0], light_position[1], closest_waypoint_ps.pose.position.x, closest_waypoint_ps.pose.position.y)
                if distance < closest_light_distance:
                    closest_light_distance = distance
                    closest_light_position = light_position

        if closest_light_position:
            state = self.get_light_state(closest_light_position)
            #return light_wp, state # TODO enable
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
