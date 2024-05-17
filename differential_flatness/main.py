#!/usr/bin/env python3

# Import libraries
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Header
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from differential_flatness import fancy_plots_3, plot_states_position, fancy_plots_4, plot_control_actions, plot_angular_velocities
import matplotlib.pyplot as plt
from rclpy.duration import Duration
from differential_flatness import create_ocp_solver
from acados_template import AcadosOcpSolver, AcadosSimSolver
from visualization_msgs.msg import Marker
import time
from scipy.spatial.transform import Rotation as R


class DifferentialFlatnessNode(Node):
    def __init__(self):
        super().__init__('planning')
        # Lets define internal variables
        self.g = 9.8
        self.mQ = 1.0

        # World axis
        self.Zw = np.array([[0.0], [0.0], [1.0]])
        self.Xw = np.array([[1.0], [0.0], [0.0]])
        self.Yw = np.array([[0.0], [1.0], [0.0]])

        # Simulation sampling time
        self.ts = 0.01
        self.t_N = 0.1
        self.timer = self.create_timer(self.ts, self.run)


        # Final time
        self.t_final = 30

        # Time Vector
        self.t = np.arange(0, self.t_final + self.ts, self.ts, dtype=np.double)

        # Inertia Matrix
        self.Jxx = 2.64e-3
        self.Jyy = 2.64e-3
        self.Jzz = 4.96e-3
        self.J = np.array([[self.Jxx, 0.0, 0.0], [0.0, self.Jyy, 0.0], [0.0, 0.0, self.Jzz]])
        self.L = [self.mQ, self.Jxx, self.Jyy, self.Jzz, self.g]

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

        self.M_d = np.zeros((3, self.t.shape[0]), dtype=np.double)
        self.f_d = np.zeros((1, self.t.shape[0]), dtype=np.double)

        # Internal States of the system        
        pos_0 = np.array([0.2, -0.2, 0.5], dtype=np.double)
        vel_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        omega_0 = np.array([0.0, 0.0, 0.0], dtype=np.double)
        quat_0 = np.array([1.0, 0.0, 0.0, 0.0])
        x_0 = np.hstack((pos_0, vel_0, quat_0, omega_0))
        self.x = np.zeros((13, self.t.shape[0] + 1), dtype=np.double)
        self.x[:, 0] = x_0

        # Publisher properties
        self.publisher_ = self.create_publisher(Odometry, "odom", 10)
        self.odometry_msg = Odometry()

        # Marker properties
        self.marker = Marker()
        self.publisher_marker_ = self.create_publisher(Marker, "ref", 10)

        # Control gains
        self.Kp = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]], dtype=np.double)
        self.Kv = np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]], dtype=np.double)
        self.kq_red  = 200
        self.kq_yaw = 3
        self.K_omega = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 8.0]], dtype=np.double)

        self.compute_reference()
        

    def send_odometry(self, x):
        self.odometry_msg.header.frame_id = "world"
        self.odometry_msg.header.stamp = self.get_clock().now().to_msg()

        self.odometry_msg.pose.pose.position.x = x[0]
        self.odometry_msg.pose.pose.position.y = x[1]
        self.odometry_msg.pose.pose.position.z = x[2]

        self.odometry_msg.pose.pose.orientation.x = x[7]
        self.odometry_msg.pose.pose.orientation.y = x[8]
        self.odometry_msg.pose.pose.orientation.z = x[9]
        self.odometry_msg.pose.pose.orientation.w = x[6]

        self.odometry_msg.twist.twist.linear.x = x[3]
        self.odometry_msg.twist.twist.linear.y = x[4]
        self.odometry_msg.twist.twist.linear.z = x[5]

        self.odometry_msg.twist.twist.angular.x = x[10]
        self.odometry_msg.twist.twist.angular.y = x[11]
        self.odometry_msg.twist.twist.angular.z = x[12]

        # Send Message
        self.publisher_.publish(self.odometry_msg)
        return None 

    def update_marker_trajectory(self, x, aux_point):
        self.marker.header.stamp = self.get_clock().now().to_msg()
        point  = Point()
        point.x = x[0]
        point.y = x[1]
        point.z = x[2]
        aux_point.append(point)
        self.marker.points = aux_point
        self.publisher_marker_.publish(self.marker)
        return aux_point

    def init_marker_trajectory(self, x):
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.marker.header.frame_id = "world"
        self.marker.type = Marker.LINE_STRIP
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.01
        self.marker.scale.y = 0.01
        self.marker.scale.z = 0.01
        self.marker.color.a = 1.0
        self.marker.color.r = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        point  = Point()
        point.x = x[0]
        point.y = x[1]
        point.z = x[2]
        aux_point = [point]
        self.marker.points = aux_point

        self.publisher_marker_.publish(self.marker)
        return aux_point
    def quatdot_c(self, quat, omega):
        # Quaternion evolution guaranteeing norm 1 (Improve this section)
        # INPUT
        # quat                                                   - actual quaternion
        # omega                                                  - angular velocities
        # OUTPUT
        # qdot                                                   - rate of change of the quaternion
        # Split values quaternion
        qw = quat[0]
        qx = quat[1]
        qy = quat[2]
        qz = quat[3]


        # Auxiliary variable in order to avoid numerical issues
        K_quat = 10
        quat_error = 1 - (qw**2 + qx**2 + qy**2 + qz**2)

        # Create skew matrix
        H_r_plus = np.array([[quat[0], -quat[1], -quat[2], -quat[3]],
                                    [quat[1], quat[0], -quat[3], quat[2]],
                                    [quat[2], quat[3], quat[0], -quat[1]],
                                    [quat[3], -quat[2], quat[1], quat[0]]])

        omega_quat = np.array([[0.0], [omega[0]], [omega[1]], [omega[2]]])


        q_dot = (1/2)*(H_r_plus@omega_quat) + K_quat*quat_error*quat
        return q_dot[:, 0]

    def desired_quaternion(self, q, omega, ts):
        # Compute the the rate of change of the quaternion
        # INPUT 
        # q                                                                                       - quaternion
        # omega                                                                                   - angular velocity
        k1 = self.quatdot_c(q, omega)
        k2 = self.quatdot_c(q+(ts/2)*k1.reshape((4,)), omega)
        k3 = self.quatdot_c(q+(ts/2)*k2.reshape((4,)), omega)
        k4 = self.quatdot_c(q+(ts)*k3.reshape((4,)), omega)
        q_k = q + (ts/6)*(k1.reshape((4,)) +2*k2.reshape((4, )) +2*k3.reshape((4, )) +k4.reshape((4, )))
        return q_k

    def compute_desired_quaternion(self, theta, theta_p, t, ts):
        # Compute the desired quaternion over time with respect to the angular velocity omega.
        # INPUT
        # theta                                                                              - desired inital angle 
        # theta_p                                                                            - angular velocity wz
        # t                                                                                  - time
        # ts                                                                                 - sample time
        # OUTPUT
        # q                                                                                  - desired quaternion 
        # Empty vector
        q = np.zeros((4, t.shape[0]), dtype = np.double)
        omega = np.zeros((3, t.shape[0]), dtype = np.double)

        # Euler angles to quaternion
        r = R.from_euler('zyx',[theta[0], 0, 0], degrees=False)
        r_q = r.as_quat()

        # Initial conditions
        q[0, 0] = r_q[3]
        q[1, 0] = r_q[0]
        q[2, 0] = r_q[1]
        q[3, 0] = r_q[2]

        # Angular velocity only z axis
        omega[2, :] = theta_p

        # Compute desired quaternion
        for k in range(0, t.shape[0]-1):
            q[:, k+1] = self.desired_quaternion(q[:, k], omega[:, k], ts)
        return  q

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

        #fig11, ax11, ax21, ax31 = fancy_plots_3()
        #plot_states_position(fig11, ax11, ax21, ax31, hd[0:3, :], hd_p[0:3, :], self.t, "Position of the System")
        #plt.show()

        #fig12, ax12, ax22, ax32 = fancy_plots_3()
        #plot_states_position(fig12, ax12, ax22, ax32, alpha[0:3, :], beta[0:3, :], self.t, "Alpha beta")
        #plt.show()

        ## Control Actions
        #fig13, ax13, ax23, ax33, ax43 = fancy_plots_4()
        #plot_control_actions(fig13, ax13, ax23, ax33, ax43, self.f_d, self.M_d, self.t, "Control Actions of the System")
        #plt.show()

        #fig14, ax14, ax24, ax34 = fancy_plots_3()
        #plot_angular_velocities(fig14, ax14, ax24, ax34, self.w_d[0:3, :], self.t, "Angular Velocities Body Frame")
        #plt.show()

        #fig15, ax15, ax25, ax35 = fancy_plots_3()
        #plot_angular_velocities(fig15, ax15, ax25, ax35, self.w_d_d[0:3, :], self.t, "Angular Accelerations Body Frame")
        #plt.show()
    def run(self):
        # Prediction Node of the NMPC formulation
        N = np.arange(0, self.t_N + self.ts, self.ts)
        N_prediction = N.shape[0]

         # Control actions
        F = np.zeros((1, self.t.shape[0]), dtype=np.double)
        M = np.zeros((3, self.t.shape[0]), dtype=np.double)

        # Generalized control actions
        u = np.zeros((4, self.t.shape[0]), dtype=np.double)


        # Constraints on control actions
        F_max = self.mQ*self.g + 20
        F_min = 0
        tau_1_max = 0.1
        tau_1_min = -0.1
        tau_2_max = 0.1
        tau_2_min = -0.1
        tau_3_max = 0.1
        taux_3_min = -0.1

        # Create optimizer
        ocp = create_ocp_solver(self.x[:, 0], N_prediction, self.t_N, F_max, F_min, tau_1_max, tau_1_min, tau_2_max, tau_2_min, tau_3_max, taux_3_min, self.L, self.ts)

        acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

        acados_integrator = AcadosSimSolver(ocp, json_file="acados_sim_" + ocp.model.name + ".json", build= True, generate= True)

        
        # Compute desired Quaternions
        # Send odometry initial conditions
        self.send_odometry(self.x[:, 0])

        # Init Trajectory
        aux_marker = self.init_marker_trajectory(self.h_d[:, 0])

        for k in range(0, self.t.shape[0]):
            # Get model
            tic = self.get_clock().now()

            # Trajectory
            aux_marker = self.update_marker_trajectory(self.h_d[:, k], aux_marker)

            # Compute Control Actions
            u[0, k], u[1:4, k] = self.position_control(self.x[0:3, k], self.h_d[0:3, k], self.x[3:6, k], self.h_d_d[0:3, k], self.h_d_dd[0:3, k], self.x[6:10, k], self.psi_d[k], self.x[10:13, k], self.w_d[:, k], self.w_d_d[:, k])
            F[:, k] = u[0, k]
            M[:, k] = u[1:4, k]
            #u[1:4, k] = self.M_d[:, k]
            # Do something
            acados_integrator.set("x", self.x[:, k])
            acados_integrator.set("u", u[:, k])

            status_integral = acados_integrator.solve()
            xcurrent = acados_integrator.get("x")

            # System evolution
            self.x[:, k+1] = xcurrent
            self.send_odometry(self.x[:, k + 1])
            self.get_logger().info("Quadrotor Simulation")

            rclpy.spin_once(self, timeout_sec = 0)
            while (self.get_clock().now() - tic) <= Duration(seconds=self.ts):
                None

        fig11, ax11, ax21, ax31 = fancy_plots_3()
        plot_states_position(fig11, ax11, ax21, ax31, self.x[0:3, :], self.h_d[0:3, :], self.t, "Position of the System")
        plt.show()

        # Control Actions
        fig13, ax13, ax23, ax33, ax43 = fancy_plots_4()
        plot_control_actions(fig13, ax13, ax23, ax33, ax43, F, M, self.t, "Control Actions of the System")
        plt.show()
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


        M_axu = self.kq_red * qe_red + self.kq_yaw*np.sign(qe_w)*qe_yaw + self.K_omega@(omega_d - omega) + omega_d_d

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
        Q = 1

        # Compute desired reference x y z
        xd = 4 * np.sin(mul * 0.04* t)
        yd = 4 * np.sin(mul * 0.08 * t)
        zd = 1 * np.sin(Q*t) + 1

        # Compute velocities
        xd_p = 4 * mul * 0.04 * np.cos(mul * 0.04 * t)
        yd_p = 4 * mul * 0.08 * np.cos(mul * 0.08 * t)
        zd_p = 1 * Q * np.cos(Q * t)

        # Compute acceleration
        xd_pp = -4 * mul * mul * 0.04 * 0.04 * np.sin(mul * 0.04 * t)
        yd_pp = -4 * mul * mul * 0.08 * 0.08 * np.sin(mul * 0.08 * t);  
        zd_pp = -1 * Q * Q *  np.sin(Q * t)

        # Compute jerk
        xd_ppp = -4 * mul * mul * mul * 0.04 * 0.04 * 0.04 * np.cos(mul * 0.04 * t)
        yd_ppp = -4 * mul * mul * mul * 0.08 * 0.08 * 0.08 * np.cos(mul * 0.08 * t);  
        zd_ppp = -1 * Q * Q * Q * np.cos(Q * t)

        # Compute snap
        xd_pppp = 4 * mul * mul * mul * mul * 0.04 * 0.04 * 0.04 * 0.04 * np.sin(mul * 0.04 * t)
        yd_pppp = 4 * mul * mul * mul * mul * 0.08 * 0.08 * 0.08 * 0.08 * np.sin(mul * 0.08 * t);  
        zd_pppp = 1 * Q * Q * Q * Q * np.sin(Q * t)

        # Compute angular displacement
        theta = np.arctan2(yd_p, xd_p)
        theta = theta

        # Compute angular velocity
        theta_p = (1. / ((yd_p / xd_p) ** 2 + 1)) * ((yd_pp * xd_p - yd_p * xd_pp) / xd_p ** 2)
        theta_p = theta_p 

        theta_pp = np.zeros((theta.shape[0]))
        theta_pp[0] = 0.0

        # Compute the angular acceleration
        for k in range(1, theta_p.shape[0]):
            theta_pp[k] = (theta_p[k] - theta_p[k-1])/self.ts
        theta_pp = theta_pp
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
    finally:
        planning_node.destroy_node()
        rclpy.shutdown()
    return None

if __name__ == '__main__':
    main()