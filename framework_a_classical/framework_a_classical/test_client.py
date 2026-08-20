"""
Test client for Framework A's ExecuteReach action server.

Three modes:
  single  -- send one target (position, optionally orientation)
  batch   -- send N GUARANTEED-REACHABLE targets, generated via forward
             kinematics from random valid joint configurations (same
             methodology used to validate the IK solver itself), report
             aggregate statistics
  file    -- load targets from a JSON file (list of {x,y,z,qx,qy,qz,qw})

Usage:
  ros2 run framework_a_classical test_client -- --mode single
  ros2 run framework_a_classical test_client -- --mode batch --n 20 --seed 1
  ros2 run framework_a_classical test_client -- --mode file --input targets.json
  ros2 run framework_a_classical test_client -- --mode batch --n 10 --output results.json
"""

import sys
import json
import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped
from benchmark_core_msgs.action import ExecuteReach

from framework_a_classical.kinematics import forward_kinematics, N_JOINTS
from framework_a_classical.ik_solver import joint_limits_array


def rotmat_to_quat(R):
    """Robust (all-branch) rotation matrix -> quaternion, Shepperd's method."""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    return float(qx), float(qy), float(qz), float(qw)


def generate_reachable_targets(n, seed=None):
    """
    Generate N guaranteed-reachable (position, quaternion) targets by sampling
    random valid joint configurations and running them through forward
    kinematics -- the SAME methodology used to validate the IK solver itself
    (94.5% success rate, 200-trial test). Ensures the client is testing real
    reachability failures, not artifacts of picking an impossible target.
    """
    lower, upper = joint_limits_array()
    rng = np.random.default_rng(seed)
    targets = []
    for _ in range(n):
        theta = rng.uniform(lower, upper)
        T, _, _ = forward_kinematics(theta)
        pos = T[:3, 3]
        qx, qy, qz, qw = rotmat_to_quat(T[:3, :3])
        targets.append({
            'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]),
            'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw,
        })
    return targets


class TestClient(Node):
    def __init__(self):
        super().__init__('framework_a_test_client')
        self._client = ActionClient(self, ExecuteReach, 'execute_reach')
        self.results = []

    def wait_for_server(self, timeout_sec=15.0):
        self.get_logger().info('Waiting for execute_reach action server...')
        available = self._client.wait_for_server(timeout_sec=timeout_sec)
        if not available:
            self.get_logger().error(
                f'Action server not available after {timeout_sec}s -- '
                'is controller_node running?')
            return False
        return True

    def send_goal(self, target, timeout=10.0, index=None, total=None):
        """target: dict with x,y,z and optionally qx,qy,qz,qw (defaults to identity)."""
        goal = ExecuteReach.Goal()
        goal.target_pose = PoseStamped()
        goal.target_pose.header.frame_id = 'base_link'
        goal.target_pose.pose.position.x = target['x']
        goal.target_pose.pose.position.y = target['y']
        goal.target_pose.pose.position.z = target['z']
        goal.target_pose.pose.orientation.x = target.get('qx', 0.0)
        goal.target_pose.pose.orientation.y = target.get('qy', 0.0)
        goal.target_pose.pose.orientation.z = target.get('qz', 0.0)
        goal.target_pose.pose.orientation.w = target.get('qw', 1.0)
        goal.timeout_seconds = timeout

        label = f'[{index+1}/{total}] ' if index is not None else ''
        self.get_logger().info(
            f"{label}Sending goal: position=({target['x']:.3f}, "
            f"{target['y']:.3f}, {target['z']:.3f})")

        send_goal_future = self._client.send_goal_async(
            goal, feedback_callback=self._feedback_cb)
        rclpy.spin_until_future_complete(self, send_goal_future, timeout_sec=timeout + 5.0)

        goal_handle = send_goal_future.result()
        if goal_handle is None:
            self.get_logger().error(f'{label}Goal send timed out (no response from server)')
            return self._record_failure(target, 'goal_send_timeout')

        if not goal_handle.accepted:
            self.get_logger().error(f'{label}Goal rejected by server')
            return self._record_failure(target, 'goal_rejected')

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout + 5.0)

        wrapped_result = result_future.result()
        if wrapped_result is None:
            self.get_logger().error(f'{label}Result retrieval timed out')
            return self._record_failure(target, 'result_timeout')

        result = wrapped_result.result
        record = {
            'target': target,
            'success': bool(result.success),
            'final_position_error': float(result.final_position_error),
            'final_orientation_error': float(result.final_orientation_error),
            'completion_time_seconds': float(result.completion_time_seconds),
            'trajectory_jerk_rms': float(result.trajectory_jerk_rms),
            'inference_latency_ms': float(result.inference_latency_ms),
            'error': None,
        }
        self.results.append(record)

        status = 'SUCCESS' if record['success'] else 'FAILED'
        self.get_logger().info(
            f"{label}{status} -- pos_err={record['final_position_error']*1000:.2f}mm  "
            f"rot_err={np.degrees(record['final_orientation_error']):.2f}deg  "
            f"time={record['completion_time_seconds']:.2f}s  "
            f"latency={record['inference_latency_ms']:.1f}ms")
        return record

    def _record_failure(self, target, error_type):
        record = {
            'target': target, 'success': False,
            'final_position_error': None, 'final_orientation_error': None,
            'completion_time_seconds': None, 'trajectory_jerk_rms': None,
            'inference_latency_ms': None, 'error': error_type,
        }
        self.results.append(record)
        return record

    def _feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().debug(
            f'  feedback: pos_error={fb.current_position_error:.4f} m, '
            f'elapsed={fb.elapsed_time_seconds:.2f}s')

    def print_summary(self):
        n = len(self.results)
        if n == 0:
            self.get_logger().warn('No results to summarize.')
            return

        successes = [r for r in self.results if r['success']]
        n_success = len(successes)

        print('\n' + '=' * 60)
        print(f'SUMMARY -- {n_success}/{n} succeeded ({100*n_success/n:.1f}%)')
        print('=' * 60)

        if successes:
            pos_errs = [r['final_position_error'] for r in successes]
            rot_errs = [np.degrees(r['final_orientation_error']) for r in successes]
            times = [r['completion_time_seconds'] for r in successes]
            latencies = [r['inference_latency_ms'] for r in successes]

            print(f"Position error (mm):    mean={np.mean(pos_errs)*1000:.3f}  "
                  f"max={np.max(pos_errs)*1000:.3f}")
            print(f"Orientation error (deg): mean={np.mean(rot_errs):.3f}  "
                  f"max={np.max(rot_errs):.3f}")
            print(f"Completion time (s):    mean={np.mean(times):.3f}  "
                  f"max={np.max(times):.3f}")
            print(f"IK latency (ms):        mean={np.mean(latencies):.2f}  "
                  f"max={np.max(latencies):.2f}")

        failures = [r for r in self.results if not r['success']]
        if failures:
            error_types = {}
            for r in failures:
                key = r['error'] or 'timeout_no_convergence'
                error_types[key] = error_types.get(key, 0) + 1
            print(f"\nFailure breakdown: {error_types}")
        print('=' * 60)

    def save_results(self, path):
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.get_logger().info(f'Results saved to {path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Framework A test client')
    parser.add_argument('--mode', choices=['single', 'batch', 'file'], default='single')
    parser.add_argument('--x', type=float, default=-0.19196833)
    parser.add_argument('--y', type=float, default=-0.36431443)
    parser.add_argument('--z', type=float, default=-0.00362096)
    parser.add_argument('--qx', type=float, default=0.9057793931094689)
    parser.add_argument('--qy', type=float, default=0.06928380975120392)
    parser.add_argument('--qz', type=float, default=0.4179684455543775)
    parser.add_argument('--qw', type=float, default=-0.008113152621429219)
    parser.add_argument('--n', type=int, default=10, help='number of targets for batch mode')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--timeout', type=float, default=10.0)
    parser.add_argument('--input', type=str, default=None, help='JSON file of targets for file mode')
    parser.add_argument('--output', type=str, default=None, help='save results to this JSON path')
    return parser.parse_args(args=[a for a in sys.argv[1:] if not a.startswith('__')])


def main(args=None):
    rclpy.init(args=args)
    parsed = parse_args()
    client = TestClient()

    try:
        if not client.wait_for_server(timeout_sec=15.0):
            client.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

        if parsed.mode == 'single':
            targets = [{'x': parsed.x, 'y': parsed.y, 'z': parsed.z,
                        'qx': parsed.qx, 'qy': parsed.qy, 'qz': parsed.qz, 'qw': parsed.qw}]
        elif parsed.mode == 'batch':
            targets = generate_reachable_targets(parsed.n, seed=parsed.seed)
        elif parsed.mode == 'file':
            if not parsed.input:
                client.get_logger().error('--input required for file mode')
                sys.exit(1)
            with open(parsed.input) as f:
                targets = json.load(f)

        total = len(targets)
        for i, target in enumerate(targets):
            client.send_goal(target, timeout=parsed.timeout, index=i, total=total)

        client.print_summary()
        if parsed.output:
            client.save_results(parsed.output)

    except KeyboardInterrupt:
        client.get_logger().info('Interrupted -- printing partial summary.')
        client.print_summary()
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
