import rospy
from pid import PID

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self):

        self.linear_controller = PID(1.0, 1.0, 1.0) # TODO params
        self.angular_controller = PID(1.0, 1.0, 1.0) # TODO params

        # TODO this is not so good
        self.last_run_time = rospy.get_time()

        # TODO: Implement
        pass

    def reset(self):
        self.linear_controller.reset()
        self.angular_controller.reset()

    def control(self, current_velocity, twist_cmd):

        time_now = rospy.get_time()
        time_elapsed = time_now - self.last_run_time
        linear_error = twist_cmd.twist.linear.x - current_velocity.twist.linear.x
        angular_error = current_velocity.twist.angular.z - twist_cmd.twist.angular.z # TODO this is not so simple, attention at 180 degrees!

        linear = self.linear_controller.step(linear_error, time_elapsed)
        angular = self.angular_controller.step(angular_error, time_elapsed)

        self.last_run_time = time_now

        throttle = linear if linear > 0.0 else 0.0
        brake = -linear if linear <= 0.0 else 0.0

        return throttle, brake, angular
