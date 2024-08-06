#!/usr/bin/env python3
# Import libraries
import rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Odometry
from mujoco_msgs.msg import Control
from geometry_msgs.msg import Point
import time
from visualization_msgs.msg import Marker
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R

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

        # Final time
        self.t_final = 31

        # Time Vector
        self.t = np.arange(0, self.t_final + self.ts, self.ts, dtype=np.double)

        # Publisher properties
        self.subscriber_ = self.create_subscription(Odometry, "odom", self.callback_get_odometry, 10)

        # Reference
        self.ref_msg = Odometry()
        self.publisher_ref_ = self.create_publisher(Odometry, "ref", 10)

        # Publisher trajectory
        self.marker_msg = Marker()
        self.points = None
        self.publisher_ref_trajectory_ = self.create_publisher(Marker, 'trajectory', 10)

        # Control
        self.control_msg = Control()
        self.publisher_control_ = self.create_publisher(Control, "cmd", 10)

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
    def init_marker(self, x):
        self.marker_msg.header.frame_id = "map"
        self.marker_msg.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg.ns = "trajectory"
        self.marker_msg.id = 0
        self.marker_msg.type = Marker.LINE_STRIP
        self.marker_msg.action = Marker.ADD
        self.marker_msg.pose.orientation.w = 1.0
        self.marker_msg.scale.x = 0.02  # Line width
        self.marker_msg.color.a = 1.0  # Alpha
        self.marker_msg.color.r = 0.0  # Red
        self.marker_msg.color.g = 1.0  # Green
        self.marker_msg.color.b = 0.0  # Blue
        point = Point()
        point.x = x[0]
        point.y = x[1]
        point.z = x[2]
        self.points = [point]
        self.marker_msg.points = self.points
        return None

    def send_marker(self, x):
        self.marker_msg.header.stamp = self.get_clock().now().to_msg()
        self.marker_msg.type = Marker.LINE_STRIP
        self.marker_msg.action = Marker.ADD
        point = Point()
        point.x = x[0]
        point.y = x[1]
        point.z = x[2]
        self.points.append(point)
        self.marker_msg.points = self.points
        self.publisher_ref_trajectory_.publish(self.marker_msg)
        return None

    def ref_circular_trajectory(self, p, w_c):
        # Compute the desired Trajecotry of the system
        # INPUT 
        # t                                                - time
        # OUTPUT
        # xd, yd, zd                                       - desired position
        # theta                                            - desired orientation
        # theta_p                                          - desired angular velocity
        t = self.t

        # COmpute Desired Positions
        xd = p * np.cos(w_c*t)
        yd = p * np.sin(w_c*t)
        zd = 0 * np.zeros((t.shape[0]))

        # Compute velocities
        xd_p = - p * w_c * np.sin(w_c * t)
        yd_p =   p * w_c * np.cos(w_c * t)
        zd_p = 0 * np.zeros((t.shape[0]))

        # Compute acceleration
        xd_pp = - p * w_c * w_c * np.cos(w_c * t)
        yd_pp = - p * w_c * w_c * np.sin(w_c * t) 
        zd_pp = 0 * np.zeros((t.shape[0]))

        # Compute jerk
        xd_ppp =  p * w_c * w_c * w_c * np.sin(w_c * t)
        yd_ppp = - p * w_c * w_c * w_c * np.cos(w_c * t) 
        zd_ppp = 0 * np.zeros((t.shape[0]))

        # Compute snap
        xd_pppp = p * w_c * w_c * w_c * w_c * np.cos(w_c * t)
        yd_pppp = p * w_c * w_c * w_c * w_c * np.sin(w_c * t)
        zd_pppp = 0 * np.zeros((t.shape[0]))

        # Compute angular displacement
        theta = 0 * np.zeros((t.shape[0]))

        # Compute angular velocity
        theta_p = 0 * np.zeros((t.shape[0]))
        #theta = np.arctan2(yd_p, xd_p)
        #theta = theta

        # Compute angular velocity
        #theta_p = (1. / ((yd_p / xd_p) ** 2 + 1)) * ((yd_pp * xd_p - yd_p * xd_pp) / xd_p ** 2)
        #theta_p[0] = 0.0

        theta_pp = 0 * np.zeros((theta.shape[0]))

        hd = np.vstack((xd, yd, zd))
        hd_p = np.vstack((xd_p, yd_p, zd_p))
        hd_pp = np.vstack((xd_pp, yd_pp, zd_pp))
        hd_ppp = np.vstack((xd_ppp, yd_ppp, zd_ppp))
        hd_pppp = np.vstack((xd_pppp, yd_pppp, zd_pppp))

        return hd, theta, hd_p, theta_p, hd_pp, hd_ppp, hd_pppp, theta_pp
    def skew_matrix(self, x):
        a1 = x[0]
        a2 = x[1]
        a3 = x[2]
        A = np.array([[0.0, -a3, a2], [a3, 0.0, -a1], [-a2, a1, 0.0]], dtype=np.double)
        return A

    def trajectory(self, p, p_d, p_dd, p_ddd):
        t = self.t
        a = np.pi/2
        r = np.zeros((3, p_d.shape[1]), dtype=np.double)
        r_d = np.zeros((3, p_d.shape[1]), dtype=np.double)
        r_dd = np.zeros((3, p_d.shape[1]), dtype=np.double)
        r_ddd = np.zeros((3, p_d.shape[1]), dtype=np.double)

        for k in range(0, p_d.shape[1]):
            w = np.array([a*np.sin(0.1*t[k]), 0.0, 0.0], dtype=np.double)
            w_d = np.array([0.1*a*np.cos(0.1*t[k]), 0.0, 0.0], dtype=np.double)
            w_dd = np.array([-0.1*0.1*a*np.sin(0.1*t[k]), 0.0, 0.0], dtype=np.double)
            w_ddd = np.array([0.1*0.1*0.1*a*np.cos(0.1*t[k]), 0.0, 0.0], dtype=np.double)

            r[:, k] = expm(self.skew_matrix(w))@p[:, k]
            r_d[:, k] = expm(self.skew_matrix(w))@(p_d[:, k] + self.skew_matrix(w_d)@p[:, k])
            r_dd[:, k] = expm(self.skew_matrix(w))@(self.skew_matrix(w_d)@self.skew_matrix(w_d)@p[:, k] + 2*self.skew_matrix(w_d)@p_d[:, k] + p_dd[:, k] + self.skew_matrix(w_dd)@p[:, k])
            r_ddd[:, k] = expm(self.skew_matrix(w))@(p_ddd[:, k] + self.skew_matrix(w_ddd)@p[:, k] + 3*self.skew_matrix(w_dd)@p_d[:, k] + 3*self.skew_matrix(w_d)@p_dd[:, k] + self.skew_matrix(w_d)@self.skew_matrix(w_d)@self.skew_matrix(w_d)@p[:, k] + 3*self.skew_matrix(w_d)@self.skew_matrix(w_d)@p_d[:, k])
        return r, r_d, r_dd, r_ddd

    def compute_flatness_states(self, hd, hd_p, hd_pp, hd_ppp, theta, theta_p):

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

        # Angular vlocity
        w = np.zeros((3, hd.shape[1]), dtype=np.double)
        

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
            q[:, k] = q[:, k]/np.linalg.norm(q[:, k])
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

        return q

    # Loop system
    def run(self):

        # Trajectory Parameters 
        p = 1
        w_c = 4

        # Compute desired Quaternions
        pd, theta, pd_p, theta_p, pd_pp, pd_ppp, pd_pppp, theta_pp = self.ref_circular_trajectory(p, w_c)
        rd, rd_p, rd_pp, rd_ppp = self.trajectory(pd, pd_p, pd_pp, pd_pppp)
        qd = self.compute_flatness_states(rd, rd_p, rd_pp, rd_ppp, theta, theta_p)


        # Init marker
        self.init_marker(rd[:, 0])

        # Simulation Loop
        for k in range(0, self.t.shape[0]):
            # Get model
            tic = time.time()
            self.send_marker(rd[:, k])
            self.send_ref(rd[:, k], qd[:, k])


            self.get_logger().info("Quadrotor Control Hopf Vibration")

            # Section to guarantee same sample times
            while (time.time() - tic <= self.ts):
                pass
            toc = time.time() - tic
            print(toc)


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