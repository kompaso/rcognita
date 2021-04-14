#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

=============================================================================

This module:

main loop for a 3-wheel kinematic robot simulation for ROS

=============================================================================

"""

from argparse import ArgumentParser

import warnings
warnings.filterwarnings('ignore')

import rospy
import threading
# import controllers
# import systems
import rcognita.systems as systems
import rcognita.controllers as controllers

import rcognita.loggers as loggers
import csv
from datetime import datetime
import os

from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Point, Twist
from math import atan2, pi
import tf.transformations as tftr
from numpy import matrix, cos, arctan2, sqrt, pi, sin, cos
from rcognita.utilities import on_key_press, gen_init_coords_angles
import numpy as np
import time as time_lib
from numpy.random import randn

#------------------------------------user settings

class Task:

    def __init__(self, mode, pose_init, pose_goal, ctrl_nominal, my_3wrobot, my_ctrl_MPC, logger = None, datafile = None):
        self.RATE = rospy.get_param('/rate', 50)
        self.lock = threading.Lock()

        # initialization
        self.pose_init = pose_init
        self.pose_goal = pose_goal
        self.system = my_3wrobot

        self.ctrl_nominal = ctrl_nominal
        self.ctrl_MPC = my_ctrl_MPC

        self.dt = 0.0
        self.time_start = 0.0

        "ROS stuff"
        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=1, latch=False)
        self.sub_odom = rospy.Subscriber("/odom", Odometry, self.odometry_callback)

        self.state = np.zeros((3))
        self.dstate = np.zeros((3))
        self.new_state = np.zeros((3))
        self.new_dstate = np.zeros((3))

        self.datafile = datafile
        self.logger = logger
        self.mode = mode

        theta_goal = self.pose_goal[2]

        self.rotation_matrix = np.array([
            [cos(theta_goal), -sin(theta_goal), 0],
            [sin(theta_goal),cos(theta_goal), 0],
            [0, 0, 1]
        ])


    def odometry_callback(self, msg):
        self.lock.acquire()

        # Read current robot state
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation


        # dim_disturb = 2
        # Dq = np.zeros(dim_disturb)
        # is_disturb = 1
        # if is_disturb:
        #     sigma_q = 1e-3 * np.ones(dim_disturb)
        #     mu_q = np.zeros(dim_disturb)
        #     tau_q = np.ones(dim_disturb)
        #
        #     for k in range(0, dim_disturb):
        #         pass
        #         Dq[k] = - tau_q[k] * ( q[k] + sigma_q[k] ) # * (randn() + mu_q[k]) )
        #
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",Dq)

        cur_rpy = tftr.euler_from_quaternion((q.x, q.y, q.z, q.w))  # roll pitch yaw
        theta = cur_rpy[2]

        dx = msg.twist.twist.linear.x
        dy = msg.twist.twist.linear.y
        omega = msg.twist.twist.angular.z

        self.state = [x, y, theta]
        self.dstate = [dx, dy, omega]

        # Make transform matrix from `robot body` frame to `goal` frame
        theta_goal = self.pose_goal[2]

        rotation_matrix = np.array([
            [cos(theta_goal), -sin(theta_goal), 0],
            [sin(theta_goal),cos(theta_goal), 0],
            [0, 0, 1]
        ])
        self.rotation_matrix = np.array([
            [cos(theta_goal), -sin(theta_goal), 0],
            [sin(theta_goal),cos(theta_goal), 0],
            [0, 0, 1]
        ])
        # print(R)

        pose_matrix = np.array([
            [self.pose_goal[0]],
            [self.pose_goal[1]],
            [0]
        ])
        # print(p)

        t_mtarix = np.block([
            [rotation_matrix, pose_matrix],
            [np.array([[0, 0, 0, 1]])]
        ])
        # print(T)

        invT = np.linalg.inv(t_mtarix)
        #print(invT)
        new_theta = theta - pose_goal[2]
        if new_theta > pi:
            new_theta -= 2 * pi
        elif new_theta < -pi:
            new_theta += 2 * pi

        # new_theta = theta - pose_goal[2]
        # if new_theta > pi:
        #     if new_theta > 2 * pi:
        #         new_theta -= 2 * pi *(new_theta // (2 * pi))
        #     else:
        #         new_theta -= 2 * pi
        # elif new_theta < -pi:
        #     if new_theta < -2* pi:
        #         new_theta += 2 * pi *(new_theta // (2 * pi))
        #     else:
        #         new_theta += 2 * pi


        # POSITION transform
        temp_pos = [x, y, 0, 1]
        self.new_state = np.dot(invT, np.transpose(temp_pos))
        self.new_state = np.array([self.new_state[0], self.new_state[1], new_theta])
        print("CURRENT POSITION", self.state)

        invR = invT[:3, :3]
        zerosR = np.zeros(invR.shape)
        invR = np.linalg.inv(rotation_matrix)
        self.new_dstate = invR.dot(np.array([dx, dy, 0]).T)
        new_omega = omega
        self.new_dstate = [self.new_dstate[0], self.new_dstate[1], new_omega]


        self.lock.release()

    def spin(self):
        # self.rotation_matrix = self.rotation_matrix
        rospy.loginfo('Task started!')
        start_time = time_lib.time()
        rate = rospy.Rate(self.RATE)


        time_prev = 0.0
        self.time_start = rospy.get_time()
        while not rospy.is_shutdown() and time_lib.time() - start_time < 300:
            t = rospy.get_time() - self.time_start
            self.t = t
            print("TIME", time_lib.time() - start_time)
            # self.dt = t - time_prev
            time_prev = t

            # Manual control
            Fman = -3
            Nman = -1
            uMan = np.array([Fman, Nman])

            velocity = Twist()
            # print("Y, hello from other side", self.new_dstate)

            u = controllers.ctrl_selector(self.t, self.new_state, uMan, self.ctrl_nominal, self.ctrl_MPC, self.mode)


            self.ctrl_MPC.upd_icost(self.new_state, u)
            r = self.ctrl_MPC.rcost(self.new_state, u)
            icost = self.ctrl_MPC.icost_val
            print("INTEGRAL COST", icost)
            # print("NEEEEEEEEW", new2u)

            #self.system.receive_action(u_to_robot)
            self.ctrl_MPC.receive_sys_state(self.new_state)
            delta = 0.1


            xCoord = self.state[0]
            yCoord = self.state[1]
            alpha = self.state[2]
            v, w = u

            vx = u[0] * cos(alpha)
            vy = u[0] * sin(alpha)

            new_vec = self.rotation_matrix.dot(np.transpose([vx, vy, 0]))
            v_new = sqrt(new_vec[0]**2 + new_vec[1]**2)

            is_disturb = False
            # print(u)
            if is_disturb:
                u += np.random.normal(0, 0.01, size=2)
                u = np.clip(u, [-0.22, -2.0], [0.22, 2.0])
            #u = [v_new, u[1]]
            self.logger.log_data_row(self.datafile, self.t, xCoord, yCoord, alpha, v_new, w, r, icost, u)

            # self.new_dstate = [self.new_dstate[0], self.new_dstate[1], self.new_dstate[5]]
            # u = self.ctrl_nominal.compute_action(self.t, self.new_state)
            #print("CONTROL", u)
            # if sqrt(self.new_state[0] ** 2 + self.new_state[1] ** 2) < delta and abs(self.new_state[2]) < delta:
            #     velocity.linear.x = 0
            #     velocity.linear.z = 0
            # else:
            # print(u)

            # print(u)
            velocity.linear.x = u[0] #u[0]
            velocity.angular.z = u[1]
            self.pub_cmd_vel.publish(velocity)

            rate.sleep()
        rospy.loginfo('Task completed!')


if __name__ == "__main__":
    rospy.init_node('task2_node')

    parser = ArgumentParser()
    parser.add_argument('--mode', type=int, default=3)
    parser.add_argument('--init_x', type=int, default=None)
    parser.add_argument('--init_y', type=int, default=None)
    parser.add_argument('--init_alpha', type=float, default=None)
    parser.add_argument('--ndots', type=int, default=25)
    parser.add_argument('--radius', type=int, default=5)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--dt_bash', type=int, default=1)
    parser.add_argument('--is_bash', type=bool, default=False)
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--Nactor', type=int, default=5)
    parser.add_argument('--pred_step_size', type=int, default=6)
    parser.add_argument('--is_log_data', type=bool, default=True)
    parser.add_argument('--is_print_sim_step', type=bool, default=False)
    parser.add_argument('--is_visualization', type=bool, default=False)

    args = parser.parse_args()


    t0 = 0
    dt = args.dt
    Vmin = -0.22
    Vmax = 0.22
    Wmin = -2
    Wmax = 2


    dim_input = 2
    dim_output = 3
    dim_state = 3
    dim_disturb = 2
    Nactor = args.Nactor
    pred_step_size = args.pred_step_size * args.dt


    model_est_stage = 2 # [s]
    model_est_period = 1*dt # [s]

    model_order = 5

    prob_noise_pow = 8

    model_est_checks = 0
    buffer_size = 200


    ctrl_bnds = np.array([[Vmin, Vmax], [Wmin, Wmax]])

    if args.init_x != None and args.init_y != None and args.init_alpha != None:
        pose_goal = [args.init_x, args.init_y, args.init_alpha]

    if args.init_alpha != None and args.init_x == None and args.init_y == None:
        x = args.radius * np.cos(args.init_alpha)
        y = args.radius * np.sin(args.init_alpha)
        pose_goal = [x, y, args.init_alpha]


    # pose_goal = [2.0, 2.0, pi]
    # m = 0
    # I = 0

    rcost_struct = 1

    R1 = np.diag([1, 10, 1, 0, 0])#np.diag([1, 100, 1, 0, 0])
    R2 = np.array([[10, 2, 1, 0, 0], [0, 10, 2, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])


    Ncritic = 50
    gamma = 1

    critic_period = 5*dt

    critic_struct_Q = 3
    critic_struct_V = 3

    is_disturb = 1

    # Disturbance
    sigma_q = 1e-3 * np.ones(dim_disturb)
    mu_q = np.zeros(dim_disturb)
    tau_q = np.ones(dim_disturb)

    # Static or dynamic controller
    is_dyn_ctrl = 0

    x0 = np.zeros(dim_state)

    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M:%S")

    data_folder = 'data'
    new_path = os.path.join(data_folder, date)

    if not os.path.exists(new_path):
        os.makedirs(new_path)
    data_folder = os.path.join(new_path, time)

    if not os.path.exists(new_path):
        os.makedirs(new_path)

    ctrl_mode = args.mode
    # pose_goal = [10.0, 10.0, pi]

    name = [data_folder, 'RLsim__', ctrl_mode, '_',dt,'_',Nactor,'_',pred_step_size,'_',round(pose_goal[0],2),'_',round(pose_goal[1],2),'_',round(pose_goal[2],2), time,'.csv']
    name_str = list(map(str, name))
    datafile = ''.join(name_str)

    with open(datafile, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['t [s]', 'x [m]', 'y [m]', 'alpha [rad]', 'v [m/s]', 'omega [rad/s]', 'r', 'int r dt'] )

    my_logger = loggers.logger_3wrobot_kinematic()

    my_3wrobot = systems.sys_3wrobot_kinematic(sys_type="diff_eqn", dim_state=dim_state, dim_input=dim_input, dim_output=dim_output, dim_disturb=dim_disturb,
                                     ctrl_bnds=np.array([[Vmin, Vmax], [Wmin, Wmax]]), is_disturb=is_disturb, pars_disturb=[sigma_q, mu_q, tau_q])

    my_ctrl_MPC = controllers.ctrl_RL_pred(dim_input, dim_output, args.mode,
                                           ctrl_bnds=ctrl_bnds,
                                          t0=t0, sampling_time=dt, Nactor=Nactor, pred_step_size=pred_step_size,
                                          sys_rhs=my_3wrobot._state_dyn, sys_out=my_3wrobot.out,
                                          x_sys=x0,
                                          prob_noise_pow = prob_noise_pow, model_est_stage=model_est_stage, model_est_period=model_est_period,
                                          buffer_size=buffer_size,
                                          model_order=model_order, model_est_checks=model_est_checks,
                                          gamma=gamma, Ncritic=Ncritic, critic_period=critic_period, critic_struct_Q=critic_struct_Q, critic_struct_V=critic_struct_V, rcost_struct=rcost_struct, rcost_pars=[R1, R2])

    my_ctrl_nominal_3wrobot = controllers.ctrl_nominal_3wrobot_NI(ctrl_bnds=ctrl_bnds, ctrl_gain=0.1, t0=0, sampling_time=dt)


    task = Task(args.mode, [0, 0, 0], pose_goal, my_ctrl_nominal_3wrobot, my_3wrobot, my_ctrl_MPC, my_logger, datafile)
    task.spin()
