"""
Damped least-squares (Levenberg-Marquardt style) inverse kinematics solver
for the custom 5-DOF AX-18A manipulator.

Uses the exactly-validated forward_kinematics() and geometric_jacobian()
from kinematics.py (max error 3.29e-11 against independent finite-difference
check) as its foundation -- NOT the DH table, which is documentation-only.

Method: iterative Newton-Raphson style pose tracking, using the damped
pseudo-inverse to remain stable near singularities:

    dtheta = J^T (J J^T + lambda^2 I)^-1 * error

where `lambda` (damping) trades a small amount of accuracy near singularities
for numerical stability. lambda is adapted based on the manipulability
measure so it stays near-zero (near-exact) when well-conditioned, and grows
only when approaching a singularity.
"""

import numpy as np
from framework_a_classical.kinematics import forward_kinematics, geometric_jacobian, N_JOINTS, JOINTS


def pose_error(T_current, T_target):
    """
    6-vector error between current and target pose: [position_error (3), orientation_error (3)].
    Orientation error uses the axis-angle (rotation vector) of the relative rotation --
    standard for iterative IK, small-error linearization.
    """
    pos_err = T_target[:3, 3] - T_current[:3, 3]

    R_err = T_target[:3, :3] @ T_current[:3, :3].T
    cos_theta = np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-8:
        rot_err = np.zeros(3)
    else:
        axis = np.array([
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1]
        ]) / (2 * np.sin(theta))
        rot_err = axis * theta

    return np.concatenate([pos_err, rot_err])


def manipulability(J):
    """
    Conditioning measure for a possibly non-square Jacobian (here 6x5: 6 task
    dimensions, 5 joints). IMPORTANT: det(J J^T) is used in textbook manipulability
    for a SQUARE or "wide" (more columns than rows) Jacobian -- but this arm's J is
    "tall" (6 rows > 5 columns), so J J^T is a 6x6 matrix built from only 5
    independent columns and is therefore ALWAYS exactly rank-deficient (det = 0),
    regardless of true conditioning. Using det(JJ^T) here silently forces maximum
    damping on every iteration, even far from any real singularity -- this was
    caught during validation (manip was reading exactly 0.0 on every single
    iteration of every trial).

    Correct approach for this shape: take the smallest singular value of J itself
    via SVD. J has at most 5 nonzero singular values (rank <= 5); the smallest of
    those genuinely reflects how close the arm is to a true singularity within its
    achievable subspace.
    """
    singular_values = np.linalg.svd(J, compute_uv=False)
    return np.min(singular_values[:N_JOINTS])


def joint_limits_array():
    """(lower, upper) arrays in radians, from the LIVE manipulator_gazebo.urdf.

    All five joints confirmed uniformly +/-2.617994 rad (~+/-150deg) -- verified
    directly against the deployed URDF's <limit> tags, not assumed. This
    replaced an earlier set (joint_2: +/-1.8, joint_3: +/-2.3, joint_4: +/-2.2)
    from a much earlier RViz-collision-testing pass that no longer matches
    what's actually in the URDF Gazebo uses -- that mismatch was a real,
    previously-undetected bug: IK was solving against artificially tight
    limits on joints 2-4, confirmed when joint_4 was observed pinning at
    2.617994 in live testing (a value outside what this function used to
    allow), not the 2.2 this function previously assumed.
    """
    lower = np.array([-2.617994, -2.617994, -2.617994, -2.617994, -2.617994])
    upper = np.array([2.617994, 2.617994, 2.617994, 2.617994, 2.617994])
    return lower, upper


def clamp_to_limits(theta):
    lower, upper = joint_limits_array()
    return np.clip(theta, lower, upper)


def solve_ik(T_target, theta_init=None, max_iters=200, pos_tol=1e-4, rot_tol=1e-3,
             lambda_max=0.05, manipulability_threshold=0.01, step_scale=1.0,
             verbose=False):
    """
    Damped least-squares IK solver.

    Args:
        T_target: 4x4 target end-effector pose (base_link frame).
        theta_init: initial joint guess (defaults to zero configuration).
        max_iters: maximum Newton-Raphson iterations.
        pos_tol: position convergence tolerance (meters).
        rot_tol: orientation convergence tolerance (radians).
        lambda_max: maximum damping factor, applied at/near singularities.
        manipulability_threshold: below this, damping ramps up.
        step_scale: scales the joint update per iteration (< 1.0 for extra stability).

    Returns:
        dict with keys: 'theta' (solution), 'converged' (bool), 'iterations',
        'final_pos_error', 'final_rot_error', 'min_manipulability' (worst point
        encountered -- useful for flagging near-singular solutions).
    """
    if theta_init is None:
        theta = np.zeros(N_JOINTS)
    else:
        theta = np.array(theta_init, dtype=float).copy()

    lower, upper = joint_limits_array()
    min_manip = np.inf
    converged = False
    iters_used = 0

    for it in range(max_iters):
        T_current, _, _ = forward_kinematics(theta)
        err = pose_error(T_current, T_target)
        pos_err_norm = np.linalg.norm(err[:3])
        rot_err_norm = np.linalg.norm(err[3:])

        if pos_err_norm < pos_tol and rot_err_norm < rot_tol:
            converged = True
            iters_used = it
            break

        J = geometric_jacobian(theta)
        manip = manipulability(J)
        min_manip = min(min_manip, manip)

        lambda_floor = 1e-3
        if manip >= manipulability_threshold:
            lam = lambda_floor
        else:
            lam = lambda_floor + (lambda_max - lambda_floor) * (1.0 - manip / manipulability_threshold)

        JJt = J @ J.T
        damped_inv = np.linalg.inv(JJt + (lam ** 2) * np.eye(6))
        dtheta = J.T @ damped_inv @ err
        dtheta = step_scale * dtheta

        at_lower = theta <= lower + 1e-9
        at_upper = theta >= upper - 1e-9
        pushing_further_negative = at_lower & (dtheta < 0)
        pushing_further_positive = at_upper & (dtheta > 0)
        dtheta[pushing_further_negative] = 0.0
        dtheta[pushing_further_positive] = 0.0

        theta = theta + dtheta
        theta = clamp_to_limits(theta)  # safety net; should rarely trigger now

        iters_used = it + 1

        if verbose:
            print(f"iter {it:3d}: pos_err={pos_err_norm:.6f}  rot_err={rot_err_norm:.6f}  "
                  f"manip={manip:.6f}  lambda={lam:.6f}")

    T_final, _, _ = forward_kinematics(theta)
    final_err = pose_error(T_final, T_target)

    return {
        'theta': theta,
        'converged': converged,
        'iterations': iters_used,
        'final_pos_error': np.linalg.norm(final_err[:3]),
        'final_rot_error': np.linalg.norm(final_err[3:]),
        'min_manipulability': min_manip,
    }


def solve_ik_multistart(T_target, n_restarts=15, seed=None, **kwargs):
    """
    Runs solve_ik from multiple random initial guesses (plus the zero pose) and
    returns the best result. A single fixed starting point (e.g. always zero)
    consistently steers the solver into the same bad local basin / limit-locked
    configuration for certain target regions -- this was observed directly
    during validation (repeated failures from a fixed zero start, some of which
    did not resolve even at 2000 iterations). Multi-start is standard practice
    for iterative IK on redundant or joint-limited chains.
    """
    lower, upper = joint_limits_array()
    rng = np.random.default_rng(seed)

    best_result = None
    starts = [np.zeros(N_JOINTS)] + [rng.uniform(lower, upper) for _ in range(n_restarts)]

    for start in starts:
        result = solve_ik(T_target, theta_init=start, **kwargs)
        if result['converged']:
            return result  # first convergence is good enough -- no need to keep searching
        if best_result is None or result['final_pos_error'] < best_result['final_pos_error']:
            best_result = result

    return best_result
