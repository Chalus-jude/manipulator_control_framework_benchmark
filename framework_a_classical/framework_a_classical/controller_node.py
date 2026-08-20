#!/usr/bin/env python3

import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from benchmark_core_msgs.action import ExecuteReach
from benchmark_core.controller_interface import ControllerInterface

from framework_a_classical.kinematics import (
    forward_kinematics,
    N_JOINTS,
)

from framework_a_classical.ik_solver import solve_ik_multistart


# ============================================================
# CONFIGURATION
# ============================================================

JOINT_NAMES = [
    'joint_1',
    'joint_2',
    'joint_3',
    'joint_4',
    'joint_5',
]

CONTROL_RATE_HZ = 100.0
DT = 1.0 / CONTROL_RATE_HZ

POSITION_TOLERANCE = 1e-3       # metres
ORIENTATION_TOLERANCE = 1e-2    # radians

MAX_JOINT_VELOCITY = 1.0        # rad/s
POSITION_GAIN = 2.0             # velocity command gain

IK_RESTARTS = 15
IK_MAX_ITERS = 200


# ============================================================
# QUATERNION -> HOMOGENEOUS TRANSFORM
# ============================================================

def pose_to_matrix(position, quat):
    """
    Convert Cartesian position + quaternion into a 4x4
    homogeneous transformation matrix.

    position:
        [x, y, z]

    quat:
        [qx, qy, qz, qw]
    """

    x, y, z, w = quat

    # Normalize quaternion
    norm = np.linalg.norm(quat)

    if norm < 1e-12:
        raise ValueError("Quaternion has near-zero magnitude.")

    x /= norm
    y /= norm
    z /= norm
    w /= norm

    R = np.array([
        [
            1.0 - 2.0 * (y*y + z*z),
            2.0 * (x*y - z*w),
            2.0 * (x*z + y*w)
        ],

        [
            2.0 * (x*y + z*w),
            1.0 - 2.0 * (x*x + z*z),
            2.0 * (y*z - x*w)
        ],

        [
            2.0 * (x*z - y*w),
            2.0 * (y*z + x*w),
            1.0 - 2.0 * (x*x + y*y)
        ]
    ])

    T = np.eye(4)

    T[:3, :3] = R
    T[:3, 3] = position

    return T


# ============================================================
# ROTATION ERROR
# ============================================================

def rotation_error(R_current, R_target):
    """
    Calculate angular orientation error in radians.

    Uses:

        R_error = R_current * R_target^T

        theta = acos((trace(R_error) - 1) / 2)
    """

    R_error = R_current @ R_target.T

    cos_theta = (np.trace(R_error) - 1.0) / 2.0

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return abs(np.arccos(cos_theta))


# ============================================================
# FRAMEWORK A CONTROLLER
# ============================================================

class FrameworkAController(ControllerInterface):
    """
    Framework A:

        Cartesian target
             |
             v
        Multi-start IK
             |
             v
        Desired joint positions
             |
             v
        Joint velocity controller
             |
             v
        Gazebo / ros2_control
             |
             v
        Joint states
             |
             +------> FK ------> Cartesian error

    IMPORTANT:

    IK is solved ONCE for each new Cartesian target.

    IK is NOT solved at 100 Hz.
    """

    def __init__(self):

        super().__init__()

        self._cached_target_key = None

        self._cached_theta_target = None

        self.last_ik_latency_ms = 0.0

        self.last_ik_converged = False


    # --------------------------------------------------------
    # COMPUTE JOINT COMMAND
    # --------------------------------------------------------

    def compute_joint_command(
        self,
        current_joint_positions,
        current_joint_velocities,
        target_position,
        target_orientation_quat,
    ):

        # ----------------------------------------------------
        # Create cache key
        # ----------------------------------------------------

        target_key = (
            tuple(np.round(target_position, 6)),
            tuple(np.round(target_orientation_quat, 6)),
        )


        # ----------------------------------------------------
        # Solve IK only if target changed
        # ----------------------------------------------------

        if target_key != self._cached_target_key:

            self._cached_target_key = target_key

            start_time = time.perf_counter()

            T_target = pose_to_matrix(
                target_position,
                target_orientation_quat
            )

            ik_result = solve_ik_multistart(
                T_target,
                n_restarts=IK_RESTARTS,
                max_iters=IK_MAX_ITERS
            )

            self.last_ik_latency_ms = (
                time.perf_counter() - start_time
            ) * 1000.0

            self.last_ik_converged = bool(
                ik_result['converged']
            )

            self._cached_theta_target = np.asarray(
                ik_result['theta'],
                dtype=float
            )

            # =================================================
            # IK DIAGNOSTIC
            # =================================================
            print(
                f"[IK DIAGNOSTIC] "
                f"target=("
                f"{target_position[0]:.3f},"
                f"{target_position[1]:.3f},"
                f"{target_position[2]:.3f}) "
                f"theta_target={self._cached_theta_target} "
                f"ik_converged={self.last_ik_converged}"
            )

            print(
                f"[IK DIAGNOSTIC] "
                f"IK latency={self.last_ik_latency_ms:.3f} ms"
            )


        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if self._cached_theta_target is None:

            return {
                'joint_command': np.zeros(N_JOINTS),
                'command_type': 'velocity',
                'ik_converged': False,
                'ik_theta_target': None,
            }


        theta_target = self._cached_theta_target


        # ----------------------------------------------------
        # Joint-space position error
        # ----------------------------------------------------

        joint_error = (
            theta_target -
            current_joint_positions
        )

        # ----------------------------------------------------
        # P controller:
        #
        # velocity = Kp * position_error
        # ----------------------------------------------------

        velocity_command = (
            POSITION_GAIN * joint_error
        )


        # ----------------------------------------------------
        # Velocity saturation
        # ----------------------------------------------------

        velocity_command = np.clip(
            velocity_command,
            -MAX_JOINT_VELOCITY,
            MAX_JOINT_VELOCITY
        )


        return {
            'joint_command': velocity_command,
            'command_type': 'velocity',
            'ik_converged': self.last_ik_converged,
            'ik_theta_target': theta_target,
        }


    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self):

        self._cached_target_key = None

        self._cached_theta_target = None

        self.last_ik_latency_ms = 0.0

        self.last_ik_converged = False


# ============================================================
# ROS 2 CONTROLLER NODE
# ============================================================

class ControllerNode(Node):

    def __init__(self):

        super().__init__(
            'framework_a_controller_node'
        )


        # ----------------------------------------------------
        # Framework A controller
        # ----------------------------------------------------

        self.controller = FrameworkAController()


        # ----------------------------------------------------
        # Joint state storage
        # ----------------------------------------------------

        self.current_positions = np.zeros(
            N_JOINTS,
            dtype=float
        )

        self.current_velocities = np.zeros(
            N_JOINTS,
            dtype=float
        )

        self._joint_state_received = False


        # ----------------------------------------------------
        # Joint state subscriber
        #
        # This is the feedback from ros2_control.
        # ----------------------------------------------------

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )


        # ----------------------------------------------------
        # VELOCITY COMMAND PUBLISHER
        #
        # IMPORTANT:
        #
        # Your ros2_controllers.yaml says:
        #
        # velocity_controller:
        #     type:
        #       forward_command_controller/
        #       ForwardCommandController
        #
        # Therefore the command topic is:
        #
        # /velocity_controller/commands
        #
        # NOT:
        #
        # /effort_controller/commands
        # ----------------------------------------------------

        self.velocity_pub = self.create_publisher(
            Float64MultiArray,
            '/velocity_controller/commands',
            10
        )


        # ----------------------------------------------------
        # Action server
        # ----------------------------------------------------

        self._action_server = ActionServer(
            self,
            ExecuteReach,
            'execute_reach',
            execute_callback=self._execute_callback,
            callback_group=ReentrantCallbackGroup()
        )


        self.get_logger().info(
            'Framework A controller node ready '
            '(IK + joint velocity control).'
        )


    # ========================================================
    # JOINT STATE CALLBACK
    # ========================================================

    def _joint_state_callback(self, msg):

        """
        /joint_states can contain joints in arbitrary order.

        Therefore we DO NOT assume:

            msg.position[0] == joint_1

        Instead we construct:

            joint name -> array index
        """

        name_to_index = {
            name: i
            for i, name in enumerate(msg.name)
        }


        try:

            positions = []

            velocities = []

            for joint_name in JOINT_NAMES:

                index = name_to_index[joint_name]

                positions.append(
                    msg.position[index]
                )

                # Some JointState messages may not contain
                # velocity data.

                if len(msg.velocity) > index:

                    velocities.append(
                        msg.velocity[index]
                    )

                else:

                    velocities.append(0.0)


            self.current_positions = np.asarray(
                positions,
                dtype=float
            )

            self.current_velocities = np.asarray(
                velocities,
                dtype=float
            )

            self._joint_state_received = True


        except (KeyError, IndexError):

            # Ignore incomplete messages.

            pass


    # ========================================================
    # ACTION EXECUTION
    # ========================================================

    def _execute_callback(self, goal_handle):

        goal = goal_handle.request


        # ----------------------------------------------------
        # Extract Cartesian position
        # ----------------------------------------------------

        target_position = np.array([
            goal.target_pose.pose.position.x,
            goal.target_pose.pose.position.y,
            goal.target_pose.pose.position.z,
        ], dtype=float)


        # ----------------------------------------------------
        # Extract quaternion
        # ----------------------------------------------------

        target_quat = np.array([
            goal.target_pose.pose.orientation.x,
            goal.target_pose.pose.orientation.y,
            goal.target_pose.pose.orientation.z,
            goal.target_pose.pose.orientation.w,
        ], dtype=float)


        # ----------------------------------------------------
        # Timeout
        # ----------------------------------------------------

        timeout = float(
            goal.timeout_seconds
        )

        if timeout <= 0.0:

            timeout = 10.0


        self.get_logger().info(
            '================================================'
        )

        self.get_logger().info(
            'New ExecuteReach goal'
        )

        self.get_logger().info(
            f'Target position: '
            f'[{target_position[0]:.4f}, '
            f'{target_position[1]:.4f}, '
            f'{target_position[2]:.4f}]'
        )

        self.get_logger().info(
            f'Target quaternion: '
            f'[{target_quat[0]:.4f}, '
            f'{target_quat[1]:.4f}, '
            f'{target_quat[2]:.4f}, '
            f'{target_quat[3]:.4f}]'
        )


        # ----------------------------------------------------
        # Reset controller
        # ----------------------------------------------------

        self.controller.reset()


        # ----------------------------------------------------
        # Target transformation
        # ----------------------------------------------------

        T_target = pose_to_matrix(
            target_position,
            target_quat
        )


        # ----------------------------------------------------
        # Timing
        # ----------------------------------------------------

        start_time = time.perf_counter()


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        jerk_samples = []

        previous_velocity = None

        final_position_error = float('inf')

        final_orientation_error = float('inf')

        success = False


        # ====================================================
        # CONTROL LOOP
        # ====================================================

        while rclpy.ok():

            elapsed = (
                time.perf_counter()
                - start_time
            )


            # ------------------------------------------------
            # Timeout
            # ------------------------------------------------

            if elapsed >= timeout:

                self.get_logger().warn(
                    'ExecuteReach timed out.'
                )

                # =================================================
                # FINAL STATE DIAGNOSTIC
                # =================================================
                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"final_joint_positions="
                    f"{self.current_positions}"
                )

                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"final_position_error="
                    f"{final_position_error:.6f} m"
                )

                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"final_orientation_error="
                    f"{final_orientation_error:.6f} rad"
                )

                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"ik_theta_target="
                    f"{self.controller._cached_theta_target}"
                )

                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"ik_converged="
                    f"{self.controller.last_ik_converged}"
                )

                break


            # ------------------------------------------------
            # Wait for joint state
            # ------------------------------------------------

            if not self._joint_state_received:

                time.sleep(DT)

                continue


            # ------------------------------------------------
            # Copy state
            #
            # Important because another thread may update
            # the arrays through the subscription callback.
            # ------------------------------------------------

            current_positions = (
                self.current_positions.copy()
            )

            current_velocities = (
                self.current_velocities.copy()
            )


            # ------------------------------------------------
            # Framework A computation
            # ------------------------------------------------

            command = (
                self.controller.compute_joint_command(
                    current_joint_positions=current_positions,
                    current_joint_velocities=current_velocities,
                    target_position=target_position,
                    target_orientation_quat=target_quat,
                )
            )


            velocity = np.asarray(
                command['joint_command'],
                dtype=float
            )


            # ------------------------------------------------
            # Ensure correct command dimension
            # ------------------------------------------------

            if velocity.shape != (N_JOINTS,):

                self.get_logger().error(
                    f'Invalid velocity command shape: '
                    f'{velocity.shape}'
                )

                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"final_joint_positions="
                    f"{self.current_positions}"
                )

                break


            # ------------------------------------------------
            # Compute jerk metric
            #
            # This is actually a discrete derivative of
            # velocity, i.e. acceleration-like metric.
            #
            # True jerk would require another derivative.
            # ------------------------------------------------

            if previous_velocity is not None:

                acceleration_estimate = (
                    velocity -
                    previous_velocity
                ) / DT

                jerk_samples.append(
                    np.linalg.norm(
                        acceleration_estimate
                    )
                )


            previous_velocity = velocity.copy()


            # ------------------------------------------------
            # Publish velocity command
            # ------------------------------------------------

            command_msg = Float64MultiArray()

            command_msg.data = velocity.tolist()

            self.velocity_pub.publish(
                command_msg
            )


            # ------------------------------------------------
            # Forward kinematics
            # ------------------------------------------------

            try:

                T_current, _, _ = (
                    forward_kinematics(
                        current_positions
                    )
                )

            except Exception as exc:

                self.get_logger().error(
                    f'Forward kinematics failed: {exc}'
                )

                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"final_joint_positions="
                    f"{self.current_positions}"
                )

                break


            # ------------------------------------------------
            # Cartesian position error
            # ------------------------------------------------

            current_position = (
                T_current[:3, 3]
            )

            final_position_error = (
                np.linalg.norm(
                    current_position -
                    target_position
                )
            )


            # ------------------------------------------------
            # Orientation error
            # ------------------------------------------------

            R_current = (
                T_current[:3, :3]
            )

            R_target = (
                T_target[:3, :3]
            )

            final_orientation_error = (
                rotation_error(
                    R_current,
                    R_target
                )
            )


            # ------------------------------------------------
            # Feedback
            # ------------------------------------------------

            feedback = ExecuteReach.Feedback()

            feedback.current_position_error = float(
                final_position_error
            )

            feedback.elapsed_time_seconds = float(
                elapsed
            )

            goal_handle.publish_feedback(
                feedback
            )


            # ------------------------------------------------
            # Console monitoring
            # ------------------------------------------------

            self.get_logger().info(
                f'error={final_position_error:.5f} m | '
                f'orientation={final_orientation_error:.5f} rad | '
                f'velocity={velocity}',
                throttle_duration_sec=1.0
            )


            # ------------------------------------------------
            # Goal reached?
            # ------------------------------------------------

            if (
                final_position_error
                < POSITION_TOLERANCE
                and
                final_orientation_error
                < ORIENTATION_TOLERANCE
            ):

                success = True

                self.get_logger().info(
                    'Target reached successfully.'
                )

                # =================================================
                # SUCCESS STATE DIAGNOSTIC
                # =================================================
                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"final_joint_positions="
                    f"{self.current_positions}"
                )

                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"final_position_error="
                    f"{final_position_error:.6f} m"
                )

                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"final_orientation_error="
                    f"{final_orientation_error:.6f} rad"
                )

                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"ik_theta_target="
                    f"{self.controller._cached_theta_target}"
                )

                print(
                    f"[FINAL STATE DIAGNOSTIC] "
                    f"ik_converged="
                    f"{self.controller.last_ik_converged}"
                )

                break


            # ------------------------------------------------
            # Maintain approximately 100 Hz
            # ------------------------------------------------

            time.sleep(DT)


        # ====================================================
        # STOP ROBOT
        # ====================================================

        stop_msg = Float64MultiArray()

        stop_msg.data = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]

        self.velocity_pub.publish(
            stop_msg
        )


        # ====================================================
        # RESULT
        # ====================================================

        result = ExecuteReach.Result()

        result.success = success

        result.final_position_error = float(
            final_position_error
        )

        result.final_orientation_error = float(
            final_orientation_error
        )

        result.completion_time_seconds = float(
            time.perf_counter() -
            start_time
        )


        if jerk_samples:

            result.trajectory_jerk_rms = float(
                np.sqrt(
                    np.mean(
                        np.square(
                            np.asarray(
                                jerk_samples
                            )
                        )
                    )
                )
            )

        else:

            result.trajectory_jerk_rms = 0.0


        result.inference_latency_ms = float(
            self.controller.last_ik_latency_ms
        )


        # ----------------------------------------------------
        # Action completion
        # ----------------------------------------------------

        goal_handle.succeed()


        self.get_logger().info(
            '================================================'
        )

        self.get_logger().info(
            f'ExecuteReach finished | '
            f'success={success} | '
            f'position_error={final_position_error:.5f} m | '
            f'orientation_error={final_orientation_error:.5f} rad | '
            f'time={result.completion_time_seconds:.3f} s | '
            f'IK latency={result.inference_latency_ms:.3f} ms'
        )

        self.get_logger().info(
            '================================================'
        )


        return result


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = ControllerNode()


    # IMPORTANT:
    #
    # The action execute callback contains a blocking loop.
    #
    # Therefore we MUST use MultiThreadedExecutor.
    #
    # Otherwise:
    #
    # execute_callback
    #       |
    #       +-- blocks executor
    #                |
    #                X
    #          joint_state_callback
    #
    # and current_positions never update.
    #

    executor = MultiThreadedExecutor(
        num_threads=4
    )

    executor.add_node(node)


    try:

        executor.spin()

    except KeyboardInterrupt:

        pass

    finally:

        # Stop robot before shutdown

        stop_msg = Float64MultiArray()

        stop_msg.data = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]

        node.velocity_pub.publish(
            stop_msg
        )

        time.sleep(0.1)

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':
    main()
