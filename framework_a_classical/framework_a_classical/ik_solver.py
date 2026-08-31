
import numpy as np
from framework_a_classical.kinematics import forward_kinematics, geometric_jacobian, N_JOINTS, JOINTS


def pose_error(T_current, T_target):
   
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

    singular_values = np.linalg.svd(J, compute_uv=False)
    return np.min(singular_values[:N_JOINTS])


def joint_limits_array():
 
    lower = np.array([-2.617994, -2.617994, -2.617994, -2.617994, -2.617994])
    upper = np.array([2.617994, 2.617994, 2.617994, 2.617994, 2.617994])
    return lower, upper


def clamp_to_limits(theta):
    lower, upper = joint_limits_array()
    return np.clip(theta, lower, upper)


def solve_ik(T_target, theta_init=None, max_iters=200, pos_tol=1e-4, rot_tol=1e-3,
             lambda_max=0.05, manipulability_threshold=0.01, step_scale=1.0,
             verbose=False):
   
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
 per = joint_limits_array()
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
