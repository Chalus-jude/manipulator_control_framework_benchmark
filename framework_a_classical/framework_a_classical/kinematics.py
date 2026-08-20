"""
Forward kinematics and geometric Jacobian for the custom 5-DOF AX-18A manipulator.


"""

import numpy as np

# --- Joint parameters, exactly as verified in manipulator.urdf ---
# Each entry: (xyz translation, rpy fixed-axis rotation, local rotation axis)
JOINTS = [
    # joint_1: base_link -> link_1
    dict(xyz=[-0.180075738834269, -0.0399304, 7.57388346011159e-05],
         rpy=[1.5707963267949, -0.0157623178412577, 0.0],
         axis=[0, 0, 1]),
    # joint_2: link_1 -> link_2
    dict(xyz=[0.0, 0.0, 0.0465],
         rpy=[3.14159265358979, 0.0105410811316586, 3.14159265358979],
         axis=[0, 1, 0]),
    # joint_3: link_2 -> link_3
    dict(xyz=[0.0, 0.0, -0.0940004988482873],
         rpy=[0.0, -0.0602056446199069, 0.0],
         axis=[0, 1, 0]),
    # joint_4: link_3 -> link_4
    dict(xyz=[0.0, 0.0, -0.0940004988482917],
         rpy=[0.0, -0.0896254408501377, 0.0],
         axis=[0, 1, 0]),
    # joint_5: link_4 -> link_5
    dict(xyz=[-0.000249682938632756, 0.000179211107190711, -0.0953824695796134],
         rpy=[-1.57079632679492, 0.0, 2.59269156911032],
         axis=[0, 1, 0]),
]

N_JOINTS = len(JOINTS)


def rpy_to_matrix(rpy):
    """Fixed-axis (extrinsic) roll-pitch-yaw to 3x3 rotation matrix: R = Rz(y) @ Ry(p) @ Rx(r)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle_to_matrix(axis, theta):
    """Rotation matrix for a rotation of `theta` about a given local axis (Rodrigues' formula)."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def joint_transform(joint, theta):
    """4x4 homogeneous transform for one joint: fixed origin, then rotation by theta about its axis."""
    R_origin = rpy_to_matrix(joint['rpy'])
    R_joint = axis_angle_to_matrix(joint['axis'], theta)
    R_total = R_origin @ R_joint

    T = np.eye(4)
    T[:3, :3] = R_total
    T[:3, 3] = joint['xyz']
    return T


def forward_kinematics(thetas):
    """
    Returns the end-effector (link_5 origin) pose as a 4x4 homogeneous transform
    in the base_link frame, plus two lists of intermediate transforms:
      - transforms[i]: pose of joint i's PARENT frame (before joint i's origin or rotation)
      - pre_rotation[i]: pose after joint i's fixed origin (xyz+rpy) but BEFORE its
        own variable rotation -- this is the correct frame for the joint's axis/position
        used in the Jacobian.
    """
    assert len(thetas) == N_JOINTS

    T = np.eye(4)
    transforms = [T.copy()]
    pre_rotation = []

    for i, joint in enumerate(JOINTS):
        R_origin = rpy_to_matrix(joint['rpy'])
        T_origin = np.eye(4)
        T_origin[:3, :3] = R_origin
        T_origin[:3, 3] = joint['xyz']

        T_pre = T @ T_origin  # after fixed origin, before this joint's own rotation
        pre_rotation.append(T_pre.copy())

        R_joint = axis_angle_to_matrix(joint['axis'], thetas[i])
        T_joint_only = np.eye(4)
        T_joint_only[:3, :3] = R_joint

        T = T_pre @ T_joint_only
        transforms.append(T.copy())

    return transforms[-1], transforms, pre_rotation


def geometric_jacobian(thetas):
    """
    Standard geometric Jacobian (6xN) for a serial revolute chain, in the base_link frame.
    Rows 0-2: linear velocity. Rows 3-5: angular velocity.
    """
    T_end, transforms, pre_rotation = forward_kinematics(thetas)
    o_end = T_end[:3, 3]

    J = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        T_pre_i = pre_rotation[i]  # frame right where joint i's own rotation happens
        R_pre_i = T_pre_i[:3, :3]
        o_i = T_pre_i[:3, 3]

        z_i = R_pre_i @ np.array(JOINTS[i]['axis'])  # joint axis expressed in base_link frame
        z_i = z_i / np.linalg.norm(z_i)

        J[0:3, i] = np.cross(z_i, o_end - o_i)
        J[3:6, i] = z_i

    return J



G = 9.81  # m/s^2


LINK_MASS = [0.064401, 0.063354, 0.063354, 0.064947, 0.016408]  # link_1..link_5
LINK_COM_LOCAL = [
    [-0.00000000259, -0.00000556, 0.00045458],   # link_1
    [-0.0000000164, 0.00000131, -0.00423158],    # link_2
    [-0.0000000164, 0.00000837, -0.00423158],    # link_3
    [-0.00026539, 0.00002496, -0.00544410],      # link_4
    [-0.00412017393846792, 0.019983014952681, 0.000118252871469049],  # link_5
]


def link_com_position(thetas, link_index):
    """
    World (base_link-frame) position of link `link_index`'s (0-based, 0=link_1)
    center of mass, at joint configuration `thetas`.
    """
    _, transforms, _ = forward_kinematics(thetas)
    T_link = transforms[link_index + 1]  # transforms[i+1] = pose of link_i's own frame
    com_local = np.append(np.array(LINK_COM_LOCAL[link_index]), 1.0)
    com_world = T_link @ com_local
    return com_world[:3]


def potential_energy(thetas):
    """Total gravitational potential energy (J) of all 5 moving links."""
    U = 0.0
    for i in range(N_JOINTS):
        com = link_com_position(thetas, i)
        U += LINK_MASS[i] * G * com[2]  # z-height * weight
    return U


def gravity_torque_numerical(thetas, eps=1e-6):
    """Gravity torque via central finite-difference on potential energy: tau_j = dU/dtheta_j."""
    tau = np.zeros(N_JOINTS)
    for j in range(N_JOINTS):
        tp = np.array(thetas, dtype=float); tp[j] += eps
        tm = np.array(thetas, dtype=float); tm[j] -= eps
        tau[j] = (potential_energy(tp) - potential_energy(tm)) / (2 * eps)
    return tau


def _link_com_jacobian(thetas, link_index):
    """3xN linear-velocity Jacobian for link `link_index`'s CoM (columns > link_index are zero)."""
    _, transforms, pre_rotation = forward_kinematics(thetas)
    com_world = link_com_position(thetas, link_index)

    J = np.zeros((3, N_JOINTS))
    for j in range(link_index + 1):  # only joints up to and including this link's own joint affect it
        T_pre_j = pre_rotation[j]
        R_pre_j = T_pre_j[:3, :3]
        o_j = T_pre_j[:3, 3]
        z_j = R_pre_j @ np.array(JOINTS[j]['axis'])
        z_j = z_j / np.linalg.norm(z_j)
        J[:, j] = np.cross(z_j, com_world - o_j)
    return J


def gravity_torque_analytical(thetas):
    """Gravity torque via virtual work: tau = sum_i J_com_i^T @ (m_i * g_vec)."""
    g_vec = np.array([0.0, 0.0, -G])
    tau = np.zeros(N_JOINTS)
    for i in range(N_JOINTS):
        J_com = _link_com_jacobian(thetas, i)
        force = LINK_MASS[i] * g_vec
        tau += J_com.T @ (-force)  # torque needed to COUNTERACT gravity (hold position)
    return tau


def gravity_torque(thetas):
    """Public entry point -- analytical method (fast). Validated utility, currently
    unused by the velocity-based control pipeline (see module comment above)."""
    return gravity_torque_analytical(thetas)


DH_TABLE = [
    # (alpha_{i-1} [rad], a_{i-1} [m], theta_offset_i [rad], d_i [m])
    (1.570796,  -0.180076, -3.125830, 0.086430),   # joint 1
    (1.570796,   0.000000,  1.560255, 0.000000),   # joint 2
    (0.000000,   0.094000, -0.060206, 0.000000),   # joint 3
    (0.000000,   0.094000,  1.481171, 0.000179),   # joint 4
    (1.570796,   0.000250,  0.000000, 0.000000),   # joint 5
]

# Fixed transform from DH frame 5 (last joint's DH frame) to the actual
# link_5/end-effector frame used by forward_kinematics() above.
DH_TOOL_TRANSFORM = np.array([
    [ 8.53098396e-01,  1.17824095e-16,  5.21750061e-01,  1.17845160e-06],
    [ 5.21750061e-01, -9.91919504e-17, -8.53098396e-01,  7.57294261e-05],
    [ 2.33040854e-17,  1.00000000e+00, -9.14526165e-17,  9.53822790e-02],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
])