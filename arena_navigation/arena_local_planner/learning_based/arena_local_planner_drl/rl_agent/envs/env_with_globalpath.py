from operator import is_
from random import randint
import gym
from gym import spaces
from gym.spaces import space
from typing import Union
from stable_baselines3.common.env_checker import check_env
import yaml
from rl_agent.utils.observation_collector import ObservationCollectorWithGP
from rl_agent.utils.reward import RewardCalculator
from rl_agent.utils.debug import timeit
from task_generator.tasks import ABSTask
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from flatland_msgs.srv import StepWorld, StepWorldRequest
import time
from .flatland_gym_env import FlatlandEnv


class FlatlandEnvGP(FlatlandEnv):
    def __init__(self, ns: str,
                 task: ABSTask,
                 robot_yaml_path: str,
                 settings_yaml_path: str,
                 reward_fnc: str,
                 is_action_space_discrete,
                 safe_dist: float,
                 goal_radius: float,
                 max_steps_per_episode,
                 train_mode: bool,
                 debug: bool):
        super().__init__(ns, task, robot_yaml_path, settings_yaml_path, reward_fnc, is_action_space_discrete, safe_dist=safe_dist,
                         goal_radius=goal_radius, max_steps_per_episode=max_steps_per_episode, train_mode=train_mode, debug=debug)
        self.observation_collector = ObservationCollectorWithGP(
            self.ns, self._laser_num_beams, self._laser_max_range, 60)
        self.observation_space = self.observation_collector.get_observation_space()
        # if action_space is discrete we add a new state
        if self._is_action_space_discrete:
            self.action_space = spaces.Discrete(
                len(self._discrete_acitons)+1)
        else:
            # if action_space is continuous, we set the range of laste value to [0,1] and use a threshold to
            # trigger the replanning.
            replan_v_range = [0, 1]
            self._replan_v_thresh = 0.5
            self.action_space = spaces.Box(
                low=np.hstack([self.action_space.low, replan_v_range[0]]), 
                high=np.hstack([self.action_space.high, replan_v_range[1]]))

    def step(self, action):
        """
        done_reasons:   0   -   exceeded max steps
                        1   -   collision with obstacle
                        2   -   goal reached
        """
        replan_needed = True
        if self._is_action_space_discrete:
            if action<len(self._discrete_acitons):
                action = self._translate_disc_action(action)
                replan_needed = False
        elif action[-1]<=self._replan_v_thresh:
            replan_needed = False

        if not replan_needed:
            self._pub_action(action)
            self._steps_curr_episode += 1
            merged_obs, obs_dict = self.observation_collector.get_observations()

            # calculate reward
            reward, reward_info = self.reward_calculator.get_reward(
                obs_dict['laser_scan'], obs_dict['goal_in_robot_frame'], 
                action=action, global_plan=obs_dict['global_plan'], robot_pose=obs_dict['robot_pose'])
            # print(f"cum_reward: {reward}")
            done = reward_info['is_done']

            # print("reward:  {}".format(reward))
            
            # info
            info = {}
            if done:
                info['done_reason'] = reward_info['done_reason']
                info['is_success'] = reward_info['is_success']
            else:
                if self._steps_curr_episode == self._max_steps_per_episode:
                    done = True
                    info['done_reason'] = 0
                    info['is_success'] = 0
        else:
            self.agent_action_pub.publish(Twist())
            curr_goal = self.observation_collector._subgoal
            # publish the goal again so that the plan manager will generate a new path
            self.pub_goal(curr_goal.x,curr_goal.y, curr_goal.theta)
            self.observation_collector.require_new_global_path()
            merged_obs, obs_dict = self.observation_collector.get_observations()
            # set the reward you want
            reward = 0.1
            done = False
            info = {}
            info['done_reason'] = 0
            info['is_success'] = 0 
        return merged_obs, reward, done, info

    def reset(self):

        # set task
        # regenerate start position end goal position of the robot and change the obstacles accordingly
        self.agent_action_pub.publish(Twist())
        if self._is_train_mode:
            self._sim_step_client()
        self.task.reset()
        self.reward_calculator.reset()
        self._steps_curr_episode = 0
        obs, _ = self.observation_collector.get_observations()
        return obs  # reward, done, info can't be included

    def pub_goal(self,x,y,theta):
        self.task.robot_manager.publish_goal(x,y,theta)