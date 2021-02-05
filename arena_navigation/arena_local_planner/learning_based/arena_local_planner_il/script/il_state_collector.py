
#Aiming his file: subscribe the supervised learning nessesary data and store them as standard files
#outline of tasks
#done: # 1. action timeslot and observation timeslot are aligned.
#done  # 2. we should subscribe the keyboard controlled robots states and save as file (csv dosen't work!!, switch to h5py)(in this file)
#done  # 3. we should write a customer PPO which will be used to train ground truth acton-states pair (in pre_training_policies.py)
#done  # 4. write a customer MLP policy which is inherited from ActorCriticPolicy used to initialize ILStateCollectorEnv.
#done  # 5. get clear how to spawn random goal and map scenarios on the rviz and design an automative process to train the whole algo. (remember to add robot odom info)
#todo  # 6. after a fixed trials and finishing supervised learning we should switch into drl training


import time
import os
import numpy as np
import yaml
import h5py

from task_generator.tasks import get_predefined_task
from rl_agent.envs.flatland_gym_env import FlatlandEnv
from task_generator.tasks import ABSTask
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped

import rospy
import rospkg

class ILStateCollectorEnv(FlatlandEnv):
    def __init__(self,
                task: ABSTask,
                robot_yaml_path: str,
                settings_yaml_path: str,
                is_action_space_discrete,
                safe_dist: float = None,
                goal_radius: float = 0.1,
                max_steps_per_episode=100):
        super(ILStateCollectorEnv, self).__init__(task, robot_yaml_path, settings_yaml_path, is_action_space_discrete)
        
        # action subscriber
        self._action_subscriber = rospy.Subscriber('cmd_vel', Twist, self._get_action_cb, queue_size=1)

        self._curr_action = 6
        self._absolute_path = rospkg.RosPack().get_path('arena_local_planner_il')
        self.reset()

    def _get_action_cb(self, data:Twist):
        '''
        mapping from cmd_vel to discrete or continuous action,
        and save as a class variable.
        self._discrete_actions is a list, each element is a dict with the keys ["name", 'linear','angular'] from flatland_gym_env.py
        '''
        linear_vel = data.linear.x
        angular_vel = data.angular.z
        current_time = rospy.get_time()
        if self._is_action_space_discrete:
            if linear_vel > (self._discrete_acitons[0]['linear'] - 0.05):
                self._curr_action = 0
            elif linear_vel < -0.05:
                self._curr_action = 1
            elif linear_vel > 0.05 and angular_vel > 0.05:
                self._curr_action = 2
            elif linear_vel > 0.05 and angular_vel < -0.05:
                self._curr_action = 3
            elif angular_vel > 1.45:
                self._curr_action = 4
            elif angular_vel < -1.45:
                self._curr_action = 5
            elif np.abs(angular_vel) < 0.05 and np.abs(linear_vel) < 0.05:
                self._curr_action = 6
            else:
                self._curr_action = self._curr_action
            
        else:
            self._curr_action = [linear_vel, angular_vel]
        
    def step(self, action):

        #todo add time alignment
        start_time = rospy.get_time()
        self._steps_curr_episode += 1
        # wait for new observations
        merged_obs, obs_dict = self.observation_collector.get_observations()
        # calculate reward (don't need reward here)
        reward, reward_info = self.reward_calculator.get_reward(
            obs_dict['laser_scan'], obs_dict['goal_in_robot_frame'])
        done = reward_info['is_done']
        # info
        if not done:
            done = self._steps_curr_episode > self._max_steps_per_episode
        info = {}
        states_time_slot = rospy.get_time()
        states_with_time = (states_time_slot, merged_obs)
        self._training_data_record(states_with_time, action)
        # rechead the goal, reset task
        if np.linalg.norm(merged_obs[-2:]-merged_obs[-5:-3]) < 1:
            self.reset()

        return states_with_time, reward, done, info # merged_obs.shape: (time_slot, numpy.ndarray(361,))

    def _training_data_record(self, states, action):
        # Record the data as numpy array in HDF5
        if self._steps_curr_episode <= 1:
            self._f_action = h5py.File(self._absolute_path + '/data/action.hdf5', "w")
            self._f_state = h5py.File(self._absolute_path + '/data/state.hdf5', "w")
        self._f_action.create_dataset(str(action[0]), data=np.array(action))
        self._f_state.create_dataset(str(states[0]), data=np.array(states[1]))

    def close_file_writer(self):

        self._f_state.close()
        self._f_action.close()
        rospy.loginfo("h5py writer have been shut down.")

    def reset(self):
        self.task.reset()
        obs, _ = self.observation_collector.get_observations()

    def get_action(self):
        current_time = rospy.get_time()
        action_with_time = (current_time, self._curr_action)
        return action_with_time



if __name__ == '__main__':

    task = get_predefined_task()
    models_folder_path = rospkg.RosPack().get_path('simulator_setup')
    arena_local_planner_drl_folder_path = rospkg.RosPack().get_path('arena_local_planner_drl')
    
    rospy.init_node('il_state_collector')

    env = ILStateCollectorEnv(task,os.path.join(models_folder_path,'robot','myrobot.model.yaml'),
                    os.path.join(arena_local_planner_drl_folder_path,'configs','default_settings.yaml'),True,
                  )

    # temporary test
    n_step = 20000
    for time_step in range(n_step):
        action = env.get_action()
        merged_obs_with_time, _, done, _ = env.step(action)
        rospy.loginfo("time_step: {2} action_time: {0} obs_time:  {1} ".format(action[0], merged_obs_with_time[0], time_step))

    env.close_file_writer()
