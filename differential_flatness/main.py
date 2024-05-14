#!/usr/bin/env python3

# Import libraries
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Header
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from differential_flatness import fancy_plots_3, plot_states_position, fancy_plots_4, plot_control_actions, plot_angular_velocities
import matplotlib.pyplot as plt

class DifferentialFlatnessNode(Node):
    def __init__(self):
        super().__init__('planning')
        # Lets define internal variables
        self.g = 9.8
        self.mQ = 1.0
        self.Zw = np.array([[0.0], [0.0], [1.0]])
        self.Xw = np.array([[1.0], [0.0], [0.0]])
        self.Yw = np.array([[0.0], [1.0], [0.0]])
        self.ts = 0.1
        self.t_final = 5
        self.t = np.arange(0, self.t_final + self.ts, self.ts, dtype=np.double)
        self.publishing_timer = 1


    def compute_reference(self):
        aux_velocity = 1

        # Desired Flat outputs
        hd, theta, hd_p, theta_p, hd_pp, hd_ppp = self.ref_trajectory_agresive(aux_velocity)

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

            
        fig11, ax11, ax21, ax31 = fancy_plots_3()
        plot_states_position(fig11, ax11, ax21, ax31, hd[0:3, :], hd_p[0:3, :], self.t, "Position of the System")
        plt.show()

        fig12, ax12, ax22, ax32 = fancy_plots_3()
        plot_states_position(fig12, ax12, ax22, ax32, alpha[0:3, :], beta[0:3, :], self.t, "Alpha beta")
        plt.show()

        # Control Actions
        fig13, ax13, ax23, ax33, ax43 = fancy_plots_4()
        plot_control_actions(fig13, ax13, ax23, ax33, ax43, f, M, self.t, "Control Actions of the System")
        plt.show()

        fig14, ax14, ax24, ax34 = fancy_plots_3()
        plot_angular_velocities(fig14, ax14, ax24, ax34, w[0:3, :], self.t, "Angular Velocities Body Frame")
        plt.show()

    
    def compute_reference_old(self):
        aux_velocity = 1

        # Desired Flat outputs
        hd, theta, hd_p, theta_p, hd_pp, hd_ppp = self.ref_trajectory_agresive(aux_velocity)

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
        

        for k in range(0, hd.shape[1]):
            # Auxiliary variables
            Zb[:, k] = (hd_pp[:, k] + self.g*self.Zw[:, 0])/(np.linalg.norm(hd_pp[:, k] + self.g*self.Zw[:, 0]))
            f[:, k] = self.mQ * np.linalg.norm(hd_pp[:, k] + self.g*self.Zw[:, 0])

            # Components Desired Orientation matrix
            Yc[:, k] = np.array([-np.sin(theta[k]), np.cos(theta[k]), 0])
            Xc[:, k] = np.array([ np.cos(theta[k]), np.sin(theta[k]), 0])
            Zc[:, k] = np.array([0.0, 0.0, 1.0])

            # Body frame that is projected to the desired orientation
            Xb[:, k] = (np.cross(Yc[:, k], Zb[:, k]))/(np.linalg.norm(np.cross(Yc[:, k], Zb[:, k])))
            Yb[:, k] = (np.cross(Zb[:, k], Xb[:, k]))

        fig11, ax11, ax21, ax31 = fancy_plots_3()
        plot_states_position(fig11, ax11, ax21, ax31, hd[0:3, :], hd_p[0:3, :], self.t, "Position of the System Old")
        plt.show()

        # Control Actions
        fig13, ax13, ax23, ax33, ax43 = fancy_plots_4()
        plot_control_actions(fig13, ax13, ax23, ax33, ax43, f, M, self.t, "Control Actions of the System Old")
        plt.show()
            
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

        # Compute angular displacement
        theta = np.arctan2(yd_p, xd_p)

        # Compute angular velocity
        theta_p = (1. / ((yd_p / xd_p) ** 2 + 1)) * ((yd_pp * xd_p - yd_p * xd_pp) / xd_p ** 2)
        theta_p[0] = 0

        
        hd = np.vstack((xd, yd, zd))
        hd_p = np.vstack((xd_p, yd_p, zd_p))
        hd_pp = np.vstack((xd_pp, yd_pp, zd_pp))
        hd_ppp = np.vstack((xd_ppp, yd_ppp, zd_ppp))

        return hd, theta, hd_p, theta_p, hd_pp, hd_ppp


def main(args=None):
    rclpy.init(args=args)
    planning_node = DifferentialFlatnessNode()
    planning_node.compute_reference()
    planning_node.compute_reference_old()
    rclpy.shutdown()
    return None

if __name__ == '__main__':
    try: 
        main()
    except(KeyboardInterrupt):
        print("Error system")
        pass
    else:
        print("Complete Execution")
        pass
    