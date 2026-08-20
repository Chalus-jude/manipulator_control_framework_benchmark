"""
ControllerInterface -- the shared contract every framework (A, B, C) implements.

Per Phase 2's architecture: the harness sends a Cartesian target end-effector
pose; each framework is responsible for its OWN pose -> joint-command step
internally (Jacobian IK for A, learned IK for B, direct policy output for C).
The harness never depends on how that mapping is computed.
"""

from abc import ABC, abstractmethod


class ControllerInterface(ABC):
    """
    Every framework's controller node must implement `compute_joint_command`.

    Input:  current joint state (positions, velocities) + target end-effector
            pose (position + quaternion orientation), both in base_link frame.
    Output: joint command -- meaning (position / velocity / torque) is
            whichever control mode is native to that framework.
    """

    @abstractmethod
    def compute_joint_command(self, current_joint_positions, current_joint_velocities,
                               target_position, target_orientation_quat):
        """
        Args:
            current_joint_positions: array-like, length N_JOINTS (radians)
            current_joint_velocities: array-like, length N_JOINTS (rad/s)
            target_position: array-like, length 3 (meters, base_link frame)
            target_orientation_quat: array-like, length 4 (x,y,z,w)

        Returns:
            dict with at minimum:
                'joint_command': array-like, length N_JOINTS
                'command_type': one of 'position', 'velocity', 'effort'
            Framework-specific extra diagnostic fields (e.g. IK convergence
            info) may be included but are not part of the required contract.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset any internal controller state (e.g. PID integrators) between episodes."""
        raise NotImplementedError
