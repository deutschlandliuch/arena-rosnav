#! /usr/bin/env python
import threading
from typing import Tuple

from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import insert
import rospy
import random
import numpy as np
from scipy.interpolate import interp1d
import time  # for debuging
import threading
# observation msgs
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose2D, PoseStamped, PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from arena_plan_msgs.msg import RobotState, RobotStateStamped
from nav_msgs.msg import Path
from scipy.interpolate import interp1d
# services
from flatland_msgs.srv import StepWorld, StepWorldRequest


# message filter
import message_filters

# for transformations
from tf.transformations import *

from gym import spaces
import numpy as np

from rl_agent.utils.debug import timeit


class ObservationCollector():
    def __init__(self, ns: str, num_lidar_beams: int, lidar_range: float):
        """ a class to collect and merge observations

        Args:
            num_lidar_beams (int): [description]
            lidar_range (float): [description]
        """
        self.ns = ns
        if ns is None or ns == "":
            self.ns_prefix = "/"
        else:
            self.ns_prefix = "/"+ns+"/"

        # define observation_space
        self.observation_space = ObservationCollector._stack_spaces((
            spaces.Box(low=0, high=lidar_range, shape=(
                num_lidar_beams,), dtype=np.float32),
            spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
            spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
        ))

        self._scan = LaserScan()
        self._robot_pose = Pose2D()
        self._robot_vel = Twist()
        self._subgoal = Pose2D()
        self._globalplan = Path()

        # message_filter subscriber: laserscan, robot_pose
        self._scan_sub = rospy.Subscriber(
            f'{self.ns_prefix}scan', LaserScan, self.callback_scan, tcp_nodelay=True)
        self._robot_state_sub = rospy.Subscriber(
            f'{self.ns_prefix}robot_state', RobotStateStamped, self.callback_robot_state, tcp_nodelay=True)
        #
        self._sub_flags = {"scan_updated": False, "robot_state_updated": False}
        self._sub_flags_con = threading.Condition()

        # topic subscriber: subgoal
        # TODO should we synchronize it with other topics
        #self._subgoal_sub = message_filters.Subscriber(f'{self.ns_prefix}subgoal', PoseStamped)
        # self._subgoal_sub.registerCallback(self.callback_subgoal)
        self._subgoal_sub = rospy.Subscriber(
            f"{self.ns_prefix}subgoal", PoseStamped, self.callback_subgoal)

        self._globalplan_sub = rospy.Subscriber(
            f'{self.ns_prefix}globalPlan', Path, self.callback_global_plan)

        # service clients
        self._is_train_mode = rospy.get_param("/train_mode")
        if self._is_train_mode:
            self._service_name_step = f'{self.ns_prefix}step_world'
            self._sim_step_client = rospy.ServiceProxy(
                self._service_name_step, StepWorld)

    def get_observation_space(self):
        return self.observation_space

    def get_observations(self):
        def all_sub_received():
            ans = True
            for k, v in self._sub_flags.items():
                if v is not True:
                    ans = False
                    break
            return ans

        def reset_sub():
            self._sub_flags = dict((k, False) for k in self._sub_flags.keys())

        if self._is_train_mode:
            self.call_service_takeSimStep()
        with self._sub_flags_con:
            while not all_sub_received():
                self._sub_flags_con.wait()  # replace it with wait for later
            reset_sub()
        # rospy.logdebug(f"Current observation takes {i} steps for Synchronization")
        #print(f"Current observation takes {i} steps for Synchronization")
        scan = self._scan.ranges.astype(np.float32)
        rho, theta = ObservationCollector._get_goal_pose_in_robot_frame(
            self._subgoal, self._robot_pose)
        merged_obs = np.hstack([scan, np.array([rho, theta])])
        obs_dict = {}
        obs_dict['laser_scan'] = scan
        obs_dict['goal_in_robot_frame'] = [rho, theta]
        obs_dict['global_plan'] = self._globalplan
        obs_dict['robot_pose'] = self._robot_pose
        return merged_obs, obs_dict

    @staticmethod
    def _get_goal_pose_in_robot_frame(goal_pos: Pose2D, robot_pos: Pose2D):
        y_relative = goal_pos.y - robot_pos.y
        x_relative = goal_pos.x - robot_pos.x
        rho = (x_relative**2+y_relative**2)**0.5
        theta = (np.arctan2(y_relative, x_relative) -
                 robot_pos.theta+4*np.pi) % (2*np.pi)-np.pi
        return rho, theta

    def call_service_takeSimStep(self):
        request = StepWorldRequest()
        try:
            response = self._sim_step_client(request)
            rospy.logdebug("step service=", response)
        except rospy.ServiceException as e:
            rospy.logdebug("step Service call failed: %s" % e)

    def callback_subgoal(self, msg_Subgoal):
        self._subgoal = self.process_subgoal_msg(msg_Subgoal)
        return

    def callback_global_plan(self, msg_global_plan):
        self._globalplan = ObservationCollector.process_global_plan_msg(
            msg_global_plan)
        return

    def callback_scan(self, msg_laserscan):
        self._scan = self.process_scan_msg(msg_laserscan)
        with self._sub_flags_con:
            self._sub_flags['scan_updated'] = True
            self._sub_flags_con.notify()

    def callback_robot_state(self, msg_robotstate):
        self._robot_pose, self._robot_vel = self.process_robot_state_msg(
            msg_robotstate)
        with self._sub_flags_con:
            self._sub_flags['robot_state_updated'] = True
            self._sub_flags_con.notify()

    def process_scan_msg(self, msg_LaserScan):
        # remove_nans_from_scan
        scan = np.array(msg_LaserScan.ranges)
        scan[np.isnan(scan)] = msg_LaserScan.range_max
        msg_LaserScan.ranges = scan
        return msg_LaserScan

    def process_robot_state_msg(self, msg_RobotStateStamped):
        state = msg_RobotStateStamped.state
        pose3d = state.pose
        twist = state.twist
        return self.pose3D_to_pose2D(pose3d), twist

    def process_pose_msg(self, msg_PoseWithCovarianceStamped):
        # remove Covariance
        pose_with_cov = msg_PoseWithCovarianceStamped.pose
        pose = pose_with_cov.pose
        return self.pose3D_to_pose2D(pose)

    def process_subgoal_msg(self, msg_Subgoal):
        pose2d = self.pose3D_to_pose2D(msg_Subgoal.pose)
        return pose2d

    @staticmethod
    def process_global_plan_msg(globalplan):
        global_plan_2d = list(map(
            lambda p: ObservationCollector.pose3D_to_pose2D(p.pose), globalplan.poses))
        global_plan_np = np.array(
            list(map(lambda p2d: [p2d.x, p2d.y], global_plan_2d)))
        return global_plan_np

    @staticmethod
    def pose3D_to_pose2D(pose3d):
        pose2d = Pose2D()
        pose2d.x = pose3d.position.x
        pose2d.y = pose3d.position.y
        quaternion = (pose3d.orientation.x, pose3d.orientation.y,
                      pose3d.orientation.z, pose3d.orientation.w)
        euler = euler_from_quaternion(quaternion)
        yaw = euler[2]
        pose2d.theta = yaw
        return pose2d

    @staticmethod
    def _stack_spaces(ss: Tuple[spaces.Box]):
        low = []
        high = []
        for space in ss:
            low.extend(space.low.tolist())
            high.extend(space.high.tolist())
        return spaces.Box(np.array(low).flatten(), np.array(high).flatten())


class ObservationCollectorWithGP(ObservationCollector):
    def __init__(self, ns: str, num_lidar_beams: int, lidar_range: float, num_points_global_path: int):
        """a new Observation collector which collect global path. 

        Args:
            ns (str): [description]
            num_lidar_beams (int): [description]
            lidar_range (float): [description]
            num_points_global_path (int): [description]
        """
        super().__init__(ns, num_lidar_beams, lidar_range)
        # add new flag for synchonization
        self._sub_flags['global_path_received'] = False
        # number of the points we want, maybe set it to 60
        self._n_gp = num_points_global_path
        self._globalplan_fixed_size = None
        # the order is 
        # [laser_data(num_lidar_beams),
        #   goal(rho),
        #   goal(theta),
        #   global_path_poses_in_robot_frame(rho),
        #   global_path_poses_in_robot_frame(theta)
        self.observation_space = ObservationCollector._stack_spaces((
            spaces.Box(low=0, high=lidar_range, shape=(
                num_lidar_beams,), dtype=np.float32),
            spaces.Box(low=0, high=10, shape=(
                1,), dtype=np.float32),
            spaces.Box(low=-np.pi, high=np.pi,
                       shape=(1,), dtype=np.float32),
            spaces.Box(low=0, high=10, shape=(
                num_points_global_path), dtype=np.float32),
            spaces.Box(low=-np.pi, high=np.pi,
                       shape=(num_points_global_path), dtype=np.float32)
        ))

    def callback_global_plan(self, msg_global_plan):
    # overwrite the one in base class for changing the flag
        with self._sub_flags_con:
            self._globalplan = ObservationCollector.process_global_plan_msg(
                msg_global_plan)
            self._globalplan_fixed_size = self.cal_global_plan_fixed_size()
            self._sub_flags['global_path_received'] = True

    def get_observations(self):
        def all_sub_received():
            ans = True
            for k, v in self._sub_flags.items():
                if v is not True:
                    ans = False
                    break
            return ans

        def reset_sub():
            self._sub_flags = dict((k, False) for k in self._sub_flags.keys())

        if self._is_train_mode:
            self.call_service_takeSimStep()
        with self._sub_flags_con:
            while not all_sub_received():
                self._sub_flags_con.wait()  # replace it with wait for later
            reset_sub()
            self._sub_flags['global_path_received'] = True
        # rospy.logdebug(f"Current observation takes {i} steps for Synchronization")
        #print(f"Current observation takes {i} steps for Synchronization")
        scan = self._scan.ranges.astype(np.float32)
        rho, theta = ObservationCollector._get_goal_pose_in_robot_frame(
            self._subgoal, self._robot_pose)
        rhos,thetas = self._get_poses_in_robot_frame(self._globalplan_fixed_size,self._robot_pose)
        merged_obs = np.hstack([scan, np.array([rho, theta]),rhos,thetas])
        obs_dict = {}
        obs_dict['laser_scan'] = scan
        obs_dict['goal_in_robot_frame'] = [rho, theta]
        obs_dict['global_plan'] = self._globalplan
        obs_dict['robot_pose'] = self._robot_pose
        return merged_obs, obs_dict

    def require_new_global_path(self):
        self._sub_flags['global_path_received'] = False

    def cal_global_plan_fixed_size(self, interpolation_method='slinear'):
        """global plan is an array with the shape (n,2). Since we want to treat the global path
        as the input, the number of the points need to be fixed. 


        Args:
            interpolation_method (str): 
        Returns:
            new_global_plan (np.ndarray) global plan with fixed size

        """
        SUPPORTED_INTERP_METHODS = ('slinear', 'quadratic', 'cubic')
        assert interpolation_method in SUPPORTED_INTERP_METHODS, \
            f"Supported interpolation methods are {SUPPORTED_INTERP_METHODS}"
        distances = np.cumsum(
            np.sqrt(np.sum(np.diff(self._globalplan, axis=0)**2, axis=1)))
        normalized_distances = distances.insert(distances, 0, 0)/distances[-1]
        f = interp1d(normalized_distances, self._globalplan,
                     kind=interpolation_method, axis=0)
        new_global_plan = f(np.linspace(0, 1, self._n_gp))
        return new_global_plan

    @staticmethod
    def _get_poses_in_robot_frame(poses: np.array, robot_pos: Pose2D):
        robot_pos_np = np.array([robot_pos.x, robot_pos.y])
        relative_poses = poses - robot_pos_np
        rhos = np.sqrt(np.sum(relative_poses**2, axis=1))
        thetas = np.mod(np.arctan2(relative_poses[:, 1], relative_poses[:, 0])
                        - robot_pos.theta+4*np.pi, 2*np.pi)-np.pi
        return rhos, thetas


if __name__ == '__main__':

    rospy.init_node('states', anonymous=True)
    print("start")

    state_collector = ObservationCollector("sim1/", 360, 10)
    i = 0
    r = rospy.Rate(100)
    while(i <= 1000):
        i = i+1
        obs = state_collector.get_observations()

        time.sleep(0.001)
