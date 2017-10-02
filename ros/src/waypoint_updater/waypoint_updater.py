#!/usr/bin/env python

import rospy
import tf

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        self.traffic_waypoint_sub = rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # Publishers
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.waypoints = None
        self.inter_waypoint_distances = None
        self.latest_pose = None
        self.current_velocity = None
        self.num_waypoints = 0
        self.closest_waypoint = 0
        self.next_red_light = None
        self.ever_received_traffic_waypoint = False

        self.MAX_VELOCITY = rospy.get_param("~max_velocity")
        self.STOP_DISTANCE = rospy.get_param("~stop_distance")
        self.LOOP = rospy.get_param("~loop")
        self.MAX_ACCEL = rospy.get_param("~max_accel")

        r = rospy.Rate(5.0)
        while not rospy.is_shutdown():
            self.calculate_and_publish_next_waypoints()
            r.sleep()

    def get_braking_distance(self, vi):
        # Use the formula for constant acceleration:
        #   df = di + (vf^2 - vi^2) / 2a
        #      = 0  + (0    - vi^2) / 2a
        return -(vi ** 2) / (2 * -self.get_accel())

    def get_lookahead_indices(self, start):
        end = start + LOOKAHEAD_WPS
        if end > self.num_waypoints:
            r = range(start, self.num_waypoints)
            if self.LOOP:
                for i in range(self.num_waypoints, end):
                    r.append(i % self.num_waypoints)
        else:
            r = range(start, end)
        return r

    def get_accel(self):
        return self.MAX_ACCEL / 4

    def get_next_velocity(self, decel, vi, ix):
        assert(vi >= 0)

        if decel:
            if vi == 0:
                return 0
            a = -self.get_accel()
        elif vi < self.MAX_VELOCITY:
            a = self.get_accel()
        else:
            return self.MAX_VELOCITY

        assert(a != 0)

        # Rearrange the formula for constant acceleration:
        #   df = di + (vf^2 - vi^2) / 2a
        #   vf = sqrt(2a(df - di) + vi^2)
        vf_squared = 2 * a * self.inter_waypoint_distances[ix] + vi ** 2
        if vf_squared == 0:
            return 0
        elif vf_squared > 0:
            vf = math.sqrt(vf_squared)
            if vf < 0.01:
                return 0
            else:
                return min(vf, self.MAX_VELOCITY)
        else:
            assert(decel)
            t_at_next_waypoint = self.inter_waypoint_distances[ix] / vi
            vf = vi - t_at_next_waypoint
            if vf < 0:
                return 0
            else:
                return min(vf, self.MAX_VELOCITY)

    # Alternate implementation for reference.
    def get_next_velocity_simple(self, decel, vi, ix):
        t_at_next_waypoint = self.inter_waypoint_distances[ix] / vi

        if decel:
            # Decelerate
            vf = vi - t_at_next_waypoint
            if vf < 0:
                vf = 0
        else:
            if vi < self.MAX_VELOCITY:
                # Accelerate
                vf = vi + t_at_next_waypoint
                if vf > self.MAX_VELOCITY:
                    vf = self.MAX_VELOCITY
            else:
                vf = self.MAX_VELOCITY

    def calculate_waypoints(self):
        wps = []
        wp_vels = []
        if (self.latest_pose and self.waypoints and self.inter_waypoint_distances):
            next_wp = self.next_waypoint(self.waypoints, self.latest_pose.pose)
            self.closest_waypoint = next_wp

            indices = self.get_lookahead_indices(next_wp)

            vi = self.current_velocity.twist.linear.x

            stop_hard = False
            stop_ix = None
            decel = False
            decel_ix = None

            for ix in indices:
                if self.next_red_light and ix <= self.next_red_light:
                    distance_to_red_light = self.distance(self.waypoints, ix, self.next_red_light)
                    if distance_to_red_light < self.STOP_DISTANCE:
                        stop_hard = True
                        if stop_ix is None:
                            stop_ix = ix
                    if distance_to_red_light < self.get_braking_distance(vi):
                        decel = True
                        if decel_ix is None:
                            decel_ix = ix

                if not self.LOOP:
                    distance_to_end = self.distance(self.waypoints, ix, self.num_waypoints - 1)
                    if distance_to_end < self.STOP_DISTANCE:
                        stop_hard = True
                        if stop_ix is None:
                            stop_ix = ix
                    if distance_to_end < self.get_braking_distance(vi):
                        decel = True
                        if decel_ix is None:
                            decel_ix = ix

                if stop_hard or (self.next_red_light and ix > self.next_red_light):
                    vf = 0
                else:
                    vf = self.get_next_velocity(decel, vi, ix)

                wp = self.waypoints[ix]
                wp.twist.twist.linear.x = vf
                wps.append(wp)
                wp_vels.append(vf)

                vi = vf

            rospy.loginfo('plan: stop_hard=[{}] decel=[{}]'.format(stop_hard, decel))
            if stop_hard:
                rospy.loginfo('plan: stop hard in {} wps'.format(stop_ix - next_wp))
            if decel:
                rospy.loginfo('plan: decel in {} wps'.format(decel_ix - next_wp))

        return wps

    def calculate_and_publish_next_waypoints(self):
        final_wps = Lane()
        final_wps.waypoints = self.calculate_waypoints()

        self.final_waypoints_pub.publish(final_wps)

    def pose_cb(self, msg):
        self.latest_pose = msg

    def current_velocity_cb(self, current_velocity):
        self.current_velocity = current_velocity

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints.waypoints
        self.num_waypoints = len(self.waypoints)
        self.base_waypoints_sub.unregister()

        if self.inter_waypoint_distances is None:
            distances = []
            for i in xrange(len(self.waypoints)):
                next_i = i + 1
                if self.LOOP and (i == (self.num_waypoints - 1)):
                    next_i = 0
                distances.append(self.distance(self.waypoints, i, next_i))
            self.inter_waypoint_distances = distances

    def traffic_cb(self, msg):
        if (msg.data >= 0):
            self.next_red_light = msg.data
        else:
            self.next_red_light = None
        self.ever_received_traffic_waypoint = True

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def find_closest_waypoint(self, waypoints, pose):

        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)#  + (a.z-b.z)**2)

        closest_len = 100000

        # no need to start from 0, instead start looking from closest wp from 'just' previous run
        if self.closest_waypoint > 20:
            closest_waypoint = self.closest_waypoint - 20#0
            next_waypoint = self.closest_waypoint -20 #0
        else:
            closest_waypoint = 0
            next_waypoint = 0

        num_waypoints = self.num_waypoints
        dist = dl(waypoints[closest_waypoint].pose.pose.position, pose.position)

        while (dist < closest_len) and (closest_waypoint < num_waypoints):
            closest_waypoint = next_waypoint
            closest_len = dist
            dist = dl(waypoints[closest_waypoint+1].pose.pose.position, pose.position)
            next_waypoint += 1

        dist_prev = dl(waypoints[closest_waypoint-1].pose.pose.position, pose.position)
        dist_curr = dl(waypoints[closest_waypoint].pose.pose.position, pose.position)
        dist_next = dl(waypoints[closest_waypoint+1].pose.pose.position, pose.position)

        #rospy.loginfo("""Waypoint dist {} {} {}""".format(dist_prev, dist_curr, dist_next))

        return closest_waypoint

    def next_waypoint(self, waypoints, pose):

        closest_waypoint = self.find_closest_waypoint(waypoints, pose)

        pose_x = pose.position.x
        pose_y = pose.position.y

        pose_orient_x = pose.orientation.x
        pose_orient_y = pose.orientation.y
        pose_orient_z = pose.orientation.z
        pose_orient_w = pose.orientation.w

        quaternion = (pose_orient_x, pose_orient_y, pose_orient_z, pose_orient_w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        pose_yaw = euler[2]

        wp = waypoints[closest_waypoint]
        wp_x = wp.pose.pose.position.x
        wp_y = wp.pose.pose.position.y

        heading = math.atan2((wp_y - pose_y),(wp_x - pose_x))

        if (pose_yaw > (math.pi/4)):
            closest_waypoint += 1

        return closest_waypoint



if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
