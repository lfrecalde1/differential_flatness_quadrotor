#!/usr/bin/env python3
# Import libraries
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Header
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from differential_flatness import fancy_plots_3, plot_states_position, fancy_plots_4, plot_control_actions_reference, plot_angular_velocities
from differential_flatness import fancy_plots_1, plot_error_norm
import matplotlib.pyplot as plt
from rclpy.duration import Duration
from differential_flatness import create_ocp_solver
from acados_template import AcadosOcpSolver, AcadosSimSolver
from visualization_msgs.msg import Marker
import time
from scipy.spatial.transform import Rotation as R
from mujoco_msgs.msg import Control
import threading


class DifferentialFlatnessNode(Node):
    def __init__(self):
        super().__init__('controller')
        # Lets define internal variables
        self.g = 9.81
        self.mQ = 1.0

        # World axis
        self.Zw = np.array([[0.0], [0.0], [1.0]])
        self.Xw = np.array([[1.0], [0.0], [0.0]])
        self.Yw = np.array([[0.0], [1.0], [0.0]])

        # Simulation sampling time
        self.ts = 0.01
        self.t_N = 0.1

        # Final time
        self.t_final = 60
        self.t_init = 5

        # Time Vector
        self.t = np.arange(0, self.t_final + self.ts, self.ts, dtype=np.double)
        self.t_aux = np.arange(0, self.t_init + self.ts, self.ts, dtype=np.double)

        # Inertia Matrix
        self.Jxx = 0.00305587
        self.Jyy = 0.00159695
        self.Jzz = 0.00159687
        self.J = np.array([[self.Jxx, 0.0, 0.0], [0.0, self.Jyy, 0.0], [0.0, 0.0, self.Jzz]])
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        self.D = np.array([[self.dx, 0.0, 0.0], [0.0, self.dy, 0.0], [0.0, 0.0, self.dz]])
        self.kh = 0.00
        self.L = [self.mQ, self.Jxx, self.Jyy, self.Jzz, self.g, self.dx, self.dy, self.dz, self.kh]

        # Nominal states 
        self.h_d = np.zeros((3, self.t.shape[0]), dtype=np.double)
        self.h_d_d = np.zeros((3, self.t.shape[0]), dtype=np.double)
        self.h_d_dd = np.zeros((3, self.t.shape[0]), dtype=np.double)
        self.h_d_ddd = np.zeros((3, self.t.shape[0]), dtype=np.double)
        self.h_d_dddd = np.zeros((3, self.t.shape[0]), dtype=np.double)

        self.psi_d = np.zeros((1, self.t.shape[0]), dtype=np.double) 
        self.psi_d_d = np.zeros((1, self.t.shape[0]), dtype=np.double) 
        self.psi_d_dd = np.zeros((1, self.t.shape[0]), dtype=np.double) 

        self.w_d = np.zeros((3, self.t.shape[0]), dtype=np.double)
        self.w_d_d = np.zeros((3, self.t.shape[0]), dtype=np.double)

        self.q_d = np.zeros((4, self.t.shape[0]), dtype=np.double)

        self.M_d = np.zeros((3, self.t.shape[0]), dtype=np.double)
        self.f_d = np.zeros((1, self.t.shape[0]), dtype=np.double)

        # Internal States of the system        
        pos_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        vel_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        omega_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        quat_0 = np.array([1.0, 0.0, 0.0, 0.0])
        self.x_0 = np.hstack((pos_0, vel_0, quat_0, omega_0))
        self.x = np.zeros((13, self.t.shape[0] + 1), dtype=np.double)
        self.x[:, 0] = self.x_0

        # Publisher properties
        self.subscriber_ = self.create_subscription(Odometry, "odom", self.callback_get_odometry, 10)

        # Reference
        self.ref_msg = Odometry()
        self.publisher_ref_ = self.create_publisher(Odometry, "ref", 10)

        # Control
        self.control_msg = Control()
        self.publisher_control_ = self.create_publisher(Control, "cmd", 10)


        # Control gains
        self.Kp = np.array([[15.0, 0.0, 0.0], [0.0, 15.0, 0.0], [0.0, 0.0, 15.0]], dtype=np.double)
        self.Kv = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]], dtype=np.double)
        self.kq_red  = 250
        self.kq_yaw = 40
        self.K_omega = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.double)

        # Max velocity and acceleration
        self.V_max = 5
        self.a_max = 7
        self.n = 1

        self.compute_reference()

        # Create a thread to run the simulation and viewer
        self.simulation_thread = threading.Thread(target=self.run)
        # Start thread for the simulation
        self.simulation_thread.start()
        

    def callback_get_odometry(self, msg):

        x = np.zeros((13, ))
        x[0] = msg.pose.pose.position.x
        x[1] = msg.pose.pose.position.y
        x[2] = msg.pose.pose.position.z

        vx_b = msg.twist.twist.linear.x
        vy_b = msg.twist.twist.linear.y
        vz_b = msg.twist.twist.linear.z

        vb = np.array([[vx_b], [vy_b], [vz_b]])

        x[10] = msg.twist.twist.angular.x
        x[11] = msg.twist.twist.angular.y
        x[12] = msg.twist.twist.angular.z

        x[7] = msg.pose.pose.orientation.x 
        x[8] = msg.pose.pose.orientation.y
        x[9] = msg.pose.pose.orientation.z
        x[6] = msg.pose.pose.orientation.w

        # Rotation inertial frame
        rotational = R.from_quat([x[7], x[8], x[9], x[6]])
        rotational_matrix = rotational.as_matrix()
        vi = rotational_matrix@vb

        x[3] = vx_b
        x[4] = vy_b
        x[5] = vz_b

        self.x_0 = x
        # Send Message
        return None 


    def send_ref(self, h, q):
        self.ref_msg.header.frame_id = "map"
        self.ref_msg.header.stamp = self.get_clock().now().to_msg()

        self.ref_msg.pose.pose.position.x = h[0]
        self.ref_msg.pose.pose.position.y = h[1]
        self.ref_msg.pose.pose.position.z = h[2]

        self.ref_msg.pose.pose.orientation.x = q[1]
        self.ref_msg.pose.pose.orientation.y = q[2]
        self.ref_msg.pose.pose.orientation.z = q[3]
        self.ref_msg.pose.pose.orientation.w = q[0]


        # Send Message
        self.publisher_ref_.publish(self.ref_msg)
        return None 

    def quat_error(self, quat, quad_d):
        qd = quad_d
        q = quat
        qd_conjugate = np.array([qd[0], -qd[1], -qd[2], -qd[3]])
        quat_d_data = qd_conjugate
        quaternion = q

        H_r_plus = np.array([[quat_d_data[0], -quat_d_data[1], -quat_d_data[2], -quat_d_data[3]], 
                             [quat_d_data[1], quat_d_data[0], -quat_d_data[3], quat_d_data[2]], 
                             [quat_d_data[2], quat_d_data[3], quat_d_data[0], -quat_d_data[1]], 
                             [quat_d_data[3], -quat_d_data[2], quat_d_data[1], quat_d_data[0]]])

        q_e_aux = H_r_plus @ quaternion

        return q_e_aux

    def quatTorot_c(self, quat):
        # Function to transform a quaternion to a rotational matrix
        # INPUT
        # quat                                                       - unit quaternion
        # OUTPUT                                     
        # R                                                          - rotational matrix

        # Normalized quaternion
        q = quat
        q = q/(q.T@q)

        # Create empty variable
        #q_hat = ca.MX.zeros(3, 3)
        #q_hat[0, 1] = -q[3]
        #q_hat[0, 2] = q[2]
        #q_hat[1, 2] = -q[1]
        #q_hat[1, 0] = q[3]
        #q_hat[2, 0] = -q[2]
        #q_hat[2, 1] = q[1]

        q0 = q[0]
        q1 = q[1]
        q2 = q[2]
        q3 = q[3]

        r11 = q0**2+q1**2-q2**2-q3**2
        r12 = 2*(q1*q2-q0*q3)
        r13 = 2*(q1*q3+q0*q2)

        r21 = 2*(q1*q2+q0*q3)
        r22 = q0**2+q2**2-q1**2-q3**2
        r23 = 2*(q2*q3-q0*q1)

        r31 = 2*(q1*q3-q0*q2)
        r32 = 2*(q2*q3+q0*q1)
        r33 = q0**2+q3**2-q1**2-q2**2

        R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        return R 

    def compute_reference(self):
        aux_velocity = 10

        # Desired Flat outputs
        hd, theta, hd_p, theta_p, hd_pp, hd_ppp, hd_pppp, theta_pp = self.ref_trajectory_agresive(aux_velocity)

        # Empty vector for the internal values

        alpha =  np.zeros((3, hd.shape[1]), dtype=np.double)
        beta =  np.zeros((3, hd.shape[1]), dtype=np.double)

        # Desired Orientation matrix
        Yc = np.zeros((3, hd.shape[1]), dtype=np.double)
        Xc = np.zeros((3, hd.shape[1]), dtype=np.double)
        Zc = np.zeros((3, hd.shape[1]), dtype=np.double)

        # Auxiliary Body frame
        Yb = np.zeros((3, hd.shape[1]), dtype=np.double)
        Xb = np.zeros((3, hd.shape[1]), dtype=np.double)
        Zb = np.zeros((3, hd.shape[1]), dtype=np.double)

        q = np.zeros((4, hd.shape[1]), dtype=np.double)

        f = np.zeros((1, hd.shape[1]), dtype=np.double)
        M = np.zeros((3, hd.shape[1]), dtype=np.double)

        # Angular vlocity
        w = np.zeros((3, hd.shape[1]), dtype=np.double)
        
        f_p = np.zeros((1, hd.shape[1]), dtype=np.double)

        # Angular acceleration
        w_p = np.zeros((3, hd.shape[1]), dtype=np.double)

        # Nominal torques of the system

        for k in range(0, hd.shape[1]):
            # Auxiliary variables
            alpha[:, k] = self.mQ*hd_pp[:, k] + self.mQ*self.g*self.Zw[:, 0]
            beta[:, k] = self.mQ*hd_pp[:, k] + self.mQ*self.g*self.Zw[:, 0]

            # Components Desired Orientation matrix
            Yc[:, k] = np.array([-np.sin(theta[k]), np.cos(theta[k]), 0])
            Xc[:, k] = np.array([ np.cos(theta[k]), np.sin(theta[k]), 0])
            Zc[:, k] = np.array([0.0, 0.0, 1.0])

            # Body frame that is projected to the desired orientation
            Xb[:, k] = (np.cross(Yc[:, k], alpha[:, k]))/(np.linalg.norm(np.cross(Yc[:, k], alpha[:, k])))
            Yb[:, k] = (np.cross(beta[:, k], Xb[:, k]))/(np.linalg.norm(np.cross(beta[:, k], Xb[:, k])))
            Zb[:, k] = np.cross(Xb[:, k], Yb[:, k])

            R_d = np.array([[Xb[0, k], Yb[0, k], Zb[0, k]], [Xb[1, k], Yb[1, k], Zb[1, k]], [Xb[2, k], Yb[2, k], Zb[2, k]]])
            r_d = R.from_matrix(R_d)
            quad_d_aux = r_d.as_quat()
            q[:, k] = np.array([quad_d_aux[3], quad_d_aux[0], quad_d_aux[1], quad_d_aux[2]])
            if k > 0:
                aux_dot = np.dot(q[:, k], q[:, k-1])
                if aux_dot < 0:
                    q[:, k] = -q[:, k]
                else:
                    q[:, k] = q[:, k]
            else:
                pass
            # Compute nominal force of the in the body frame
            f[:, k] = np.dot(Zb[:, k], self.mQ*hd_pp[:, k] + self.mQ*self.g*self.Zw[:, 0])

            
            # Compute angular velocities
            # Elements of the vecto b
            b1 = self.mQ*np.dot(Xb[:, k], hd_ppp[:, k])
            b2 = -self.mQ*np.dot(Yb[:, k], hd_ppp[:, k])
            b3 = theta_p[k] * np.dot(Xc[:, k], Xb[:, k])

            b = np.array([[b1], [b2], [b3]], dtype=np.double)

            # Elements of the matrix A
            a11 = 0.0
            a12 = f[:, k]
            a13 = 0.0

            a21 = f[:, k]
            a22 = 0.0
            a23 = 0.0
            
            a31 = 0.0
            a32 = -np.dot(Yc[:, k], Zb[:, k])
            a33 = np.linalg.norm(np.cross(Yc[:, k], Zb[:, k]))

            # Inverse Matrix A
            A = np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]], dtype=np.double)
            A_1 = np.linalg.inv(A)

            # Compute nominal angular velocity
            aux_angular_velocity = A_1@b
            w[:, k] = aux_angular_velocity[:, 0]
            wx = w[0, k]
            wy = w[1, k]
            wz = w[2, k]

            # Time derivative of the force respect with the body axis
            f_p[:, k] = self.mQ*np.dot(Zb[:, k], hd_ppp[:, k])

            chi_1 = theta_pp[k] * np.dot(Xc[:, k], Xb[:, k])
            chi_2 = -2*theta_p[k] * wy * np.dot(Xc[:, k], Zb[:, k])
            chi_3 =  -wy * wx * np.dot(Yc[:, k], Yb[:, k])
            chi_4 =  2* theta_p[k] * wz * np.dot(Xc[:, k], Yb[:, k])
            chi_5 =  -wz*wx*np.dot(Yc[:, k], Zb[:, k])

            chi = chi_1 + chi_2 + chi_3 + chi_4 + chi_5

            # Compute angular accelerations of the system
            B1 = self.mQ*np.dot(Xb[:, k], hd_pppp[:, k]) - f[:, k]*wx*wz - 2*f_p[:, k]*wy
            B2 = -self.mQ*np.dot(Yb[:, k], hd_pppp[:, k]) -2 * f_p[:, k] * wx + f[:, k]*wy*wz
            B3 = chi

            B = np.array([[B1], [B2], [B3]], dtype=np.double)

            # Computing angular acceleration
            aux_angular_acce = A_1@B
            w_p[:, k] = aux_angular_acce[:, 0]
            aux_torque = self.J@w_p[:, k] + np.cross(w[:, k], self.J@w[:, k])
            # Compute torque
            M[:, k] = aux_torque


        # Updates nominal states from the differential flatness properties
        self.M_d = M            
        self.f_d = f            

        self.w_d = w
        self.w_d_d =w_p 

        self.h_d = hd
        self.h_d_d = hd_p
        self.h_d_dd = hd_pp
        self.h_d_ddd = hd_ppp
        self.h_d_dddd = hd_pppp

        self.psi_d = theta
        self.psi_d_d = theta_p
        self.psi_d_dd = theta_pp

        self.q_d = q

    def run(self):
        # Prediction Node of the NMPC formulation
        N = np.arange(0, self.t_N + self.ts, self.ts)
        N_prediction = N.shape[0]

         # Control actions
        F = np.zeros((1, self.t.shape[0]), dtype=np.double)
        M = np.zeros((3, self.t.shape[0]), dtype=np.double)

        # Generalized control actions
        u = np.zeros((4, self.t.shape[0]), dtype=np.double)

        for k in range(0, self.t_aux.shape[0]):
            tic = time.time()
            self.x[:, 0] = self.x_0
            while (time.time() - tic <= self.ts):
                pass

        # Compute desired Quaternions
        self.send_ref(self.h_d[:, 0], self.q_d[:, 0])

        h_e = np.zeros((1, self.t.shape[0]), dtype=np.double)

        for k in range(0, self.t.shape[0]):
            # Get model
            tic = time.time()

            # Send message with desired trayectory
            self.send_ref(self.h_d[:, k], self.q_d[:, k])
            h_e[:, k] = np.linalg.norm(self.x[0:3, k] - self.h_d[0:3, k])

            # Compute Control Actions
            u[0, k], u[1:4, k] = self.position_control(self.x[0:3, k], self.h_d[0:3, k], self.x[3:6, k], self.h_d_d[0:3, k], self.h_d_dd[0:3, k], self.x[6:10, k], self.psi_d[k], self.x[10:13, k], self.w_d[:, k], self.w_d_d[:, k])
            self.send_control_value(u[:, k])
            F[:, k] = u[0, k]
            M[:, k] = u[1:4, k]

            # System evolution
            self.x[:, k+1] = self.x_0

            self.get_logger().info("Quadrotor Simulation")

            # Section to guarantee same sample times
            while (time.time() - tic <= self.ts):
                pass
            toc = time.time() - tic
            print(toc)

        # Set Control action to hover
        for k in range(0, self.t_aux.shape[0]):
            tic = time.time()
            hover = np.array([self.mQ * self.g, 0, 0, 0])
            self.send_control_value(hover)
            # Section to guarantee same sample times
            while (time.time() - tic <= self.ts):
                pass
        
        # Results of the system
        fig11, ax11, ax21, ax31 = fancy_plots_3()
        plot_states_position(fig11, ax11, ax21, ax31, self.x[0:3, :], self.h_d[0:3, :], self.t, "Position of the System No drag")
        plt.show()

        # Control Actions
        fig13, ax13, ax23, ax33, ax43 = fancy_plots_4()
        plot_control_actions_reference(fig13, ax13, ax23, ax33, ax43, F, M, self.f_d, self.M_d, self.t, "Control Actions of the System No Drag")
        plt.show()

        fig14, ax14 = fancy_plots_1()
        plot_error_norm(fig14, ax14, h_e, self.t, "Error Norm of the System No Drag")
        plt.show()
        return None

    def send_control_value(self, u):
        self.control_msg.thrust = u[0]
        self.control_msg.torque_x = u[1]
        self.control_msg.torque_y = u[2]
        self.control_msg.torque_z = u[3]
        self.publisher_control_.publish(self.control_msg)
        return None

    def position_control(self, x, xd, x_d, xd_d, xd_dd, quat, psi, omega, omega_d, omega_d_d):
        kp = self.Kp
        kv = self.Kv
        aux_variable = kp@(xd - x) + kv@(xd_d - x_d) + xd_dd
        force_zb = self.mQ * (aux_variable + self.g*self.Zw[:, 0])

        

        R_b = self.quatTorot_c(quat)
        Zb = R_b[:, 2]

        force = np.dot(Zb, force_zb)

        Zb_d = force_zb/(np.linalg.norm(force_zb))
        Xc_d = np.array([ np.cos(psi), np.sin(psi), 0])
        Yb_d = (np.cross(Zb_d, Xc_d))/(np.linalg.norm(np.cross(Zb_d, Xc_d)))
        Xb_d = np.cross(Yb_d, Zb_d)

        R_d = np.array([[Xb_d[0], Yb_d[0], Zb_d[0]], [Xb_d[1], Yb_d[1], Zb_d[1]], [Xb_d[2], Yb_d[2], Zb_d[2]]])
        r_d = R.from_matrix(R_d)
        quad_d_aux = r_d.as_quat()
        quad_d = np.array([quad_d_aux[3], quad_d_aux[0], quad_d_aux[1], quad_d_aux[2]])

        quat_error = self.quat_error(quad_d, quat)

        qe_w = quat_error[0]
        qe_x = quat_error[1]
        qe_y = quat_error[2]
        qe_z = quat_error[3]

        qe_red = (1/(qe_w**2 + qe_z**2))*np.array([qe_w*qe_x-qe_y*qe_z, qe_w*qe_y + qe_x*qe_z, 0])
        qe_yaw = (1/(qe_w**2 + qe_z**2))*np.array([0, 0, qe_z])


        M_axu = self.kq_red * qe_red + self.kq_yaw*np.sign(qe_w)*qe_yaw + self.K_omega@(omega_d - omega)

        M = self.J@M_axu + np.cross(omega, self.J@omega)

        return force, M
    def ref_trajectory_agresive(self, mul):
        # Compute the desired Trajecotry of the system
        # INPUT 
        # t                                                - time
        # OUTPUT
        # xd, yd, zd                                       - desired position
        # theta                                            - desired orientation
        # theta_p                                          - desired angular velocity
        t = self.t
        r_max = (self.V_max**2)/self.a_max
        k = self.a_max/self.V_max
        r_min = (r_max)/self.n
        Q = mul

        xd = 2 * np.sin(mul * 0.04* t)
        yd = 2 * np.sin(mul * 0.08 * t)
        zd = 1 * np.sin(0.1*Q*t) + 4

        # Compute velocities
        # Compute velocities
        xd_p = 2 * mul * 0.04 * np.cos(mul * 0.04 * t)
        yd_p = 2 * mul * 0.08 * np.cos(mul * 0.08 * t)
        zd_p = 0.1 * Q * np.cos(0.1*Q * t)

        # Compute acceleration
        xd_pp = -2 * mul * mul * 0.04 * 0.04 * np.sin(mul * 0.04 * t)
        yd_pp = -2 * mul * mul * 0.08 * 0.08 * np.sin(mul * 0.08 * t);  
        zd_pp = -0.1 * 0.1 * Q * Q *  np.sin(0.1*Q * t)

        # Compute jerk
        xd_ppp = -2 * mul * mul * mul * 0.04 * 0.04 * 0.04 * np.cos(mul * 0.04 * t)
        yd_ppp = -2 * mul * mul * mul * 0.08 * 0.08 * 0.08 * np.cos(mul * 0.08 * t);  
        zd_ppp = -0.1 * 0.1 * 0.1* Q * Q * Q * np.cos(0.1*Q * t)

        # Compute snap
        xd_pppp = 2 * mul * mul * mul * mul * 0.04 * 0.04 * 0.04 * 0.04 * np.sin(mul * 0.04 * t)
        yd_pppp = 2 * mul * mul * mul * mul * 0.08 * 0.08 * 0.08 * 0.08 * np.sin(mul * 0.08 * t);  
        zd_pppp = 0.1 * 0.1 * 0.1 * 0.1 * Q * Q * Q * Q * np.sin(0.1*Q * t)

        # Compute angular displacement
        theta = np.arctan2(yd_p, xd_p)
        theta = theta

        # Compute angular velocity
        theta_p = (1. / ((yd_p / xd_p) ** 2 + 1)) * ((yd_pp * xd_p - yd_p * xd_pp) / xd_p ** 2)
        theta_p[0] = 0.0

        theta_pp = np.zeros((theta.shape[0]))
        theta_pp[0] = 0.0

        # Compute the angular acceleration
        for k in range(1, theta_p.shape[0]):
            theta_pp[k] = (theta_p[k] - theta_p[k-1])/self.ts
        hd = np.vstack((xd, yd, zd))
        hd_p = np.vstack((xd_p, yd_p, zd_p))
        hd_pp = np.vstack((xd_pp, yd_pp, zd_pp))
        hd_ppp = np.vstack((xd_ppp, yd_ppp, zd_ppp))
        hd_pppp = np.vstack((xd_pppp, yd_pppp, zd_pppp))

        return hd, theta, hd_p, theta_p, hd_pp, hd_ppp, hd_pppp, theta_pp



def main(args=None):
    rclpy.init(args=args)
    planning_node = DifferentialFlatnessNode()
    try:
        rclpy.spin(planning_node)  # Will run until manually interrupted
    except KeyboardInterrupt:
        planning_node.get_logger().info('Simulation stopped manually.')
        planning_node.destroy_node()
        rclpy.shutdown()
    finally:
        planning_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()