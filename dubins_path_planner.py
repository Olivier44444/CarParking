import math

def angle_mod(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi

def dubins_LSL(alpha, beta, d):
    tmp0 = d + math.sin(alpha) - math.sin(beta)
    tmp1 = math.cos(beta) - math.cos(alpha)
    p_squared = tmp0 * tmp0 + tmp1 * tmp1
    if p_squared < 0:
        return None
    t = angle_mod(math.atan2(tmp1, tmp0) - alpha)
    p = math.sqrt(p_squared)
    q = angle_mod(beta - math.atan2(tmp1, tmp0))
    return t, p, q

def dubins_RSR(alpha, beta, d):
    tmp0 = d - math.sin(alpha) + math.sin(beta)
    tmp1 = math.cos(alpha) - math.cos(beta)
    p_squared = tmp0 * tmp0 + tmp1 * tmp1
    if p_squared < 0:
        return None
    t = angle_mod(alpha - math.atan2(tmp1, tmp0))
    p = math.sqrt(p_squared)
    q = angle_mod(-beta + math.atan2(tmp1, tmp0))
    return t, p, q

def dubins_LSR(alpha, beta, d):
    tmp0 = d - math.sin(alpha) - math.sin(beta)
    tmp2 = 2 + d*d - 2*math.cos(alpha - beta) + 2*d*(math.sin(alpha) + math.sin(beta))
    if tmp2 < 0:
        return None
    p = math.sqrt(tmp2)
    tmp1 = math.atan2(-math.cos(alpha) - math.cos(beta), tmp0) - math.atan2(-2.0, p)
    t = angle_mod(tmp1 - alpha)
    q = angle_mod(tmp1 - angle_mod(beta))
    return t, p, q

def dubins_RSL(alpha, beta, d):
    tmp0 = d + math.sin(alpha) + math.sin(beta)
    tmp2 = 2 + d*d - 2*math.cos(alpha - beta) + 2*d*(-math.sin(alpha) - math.sin(beta))
    if tmp2 < 0:
        return None
    p = math.sqrt(tmp2)
    tmp1 = math.atan2(math.cos(alpha) + math.cos(beta), tmp0) - math.atan2(2.0, p)
    t = angle_mod(alpha - tmp1)
    q = angle_mod(beta - tmp1)
    return t, p, q

def dubins_RLR(alpha, beta, d):
    tmp0 = (6. - d*d + 2*math.cos(alpha - beta) + 2*d*(math.sin(alpha) - math.sin(beta))) / 8.
    if abs(tmp0) > 1:
        return None
    p = angle_mod(2*math.pi - math.acos(tmp0))
    t = angle_mod(alpha - math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta)) + p/2.)
    q = angle_mod(alpha - beta - t + p)
    return t, p, q

def dubins_LRL(alpha, beta, d):
    tmp0 = (6. - d*d + 2*math.cos(alpha - beta) + 2*d*(-math.sin(alpha) + math.sin(beta))) / 8.
    if abs(tmp0) > 1:
        return None
    p = angle_mod(2*math.pi - math.acos(tmp0))
    t = angle_mod(-alpha - math.atan2(math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta)) + p/2.)
    q = angle_mod(beta - alpha - t + p)
    return t, p, q

def dubins_path_planner(q0, q1, radius):
    """
    q0, q1: [x, y, yaw] initial and goal poses
    radius: turning radius
    Returns:
        px, py, pyaw: empty lists (trajectory points can be added later)
        path_type: string
        lengths: list of segment lengths [t, p, q] scaled by radius
    """
    x0, y0, yaw0 = q0
    x1, y1, yaw1 = q1

    dx = x1 - x0
    dy = y1 - y0
    D = math.hypot(dx, dy)
    d = D / radius

    theta = angle_mod(math.atan2(dy, dx))
    alpha = angle_mod(yaw0 - theta)
    beta = angle_mod(yaw1 - theta)

    # Try all path types
    paths = []

    LSL = dubins_LSL(alpha, beta, d)
    if LSL is not None:
        paths.append(('LSL', LSL))

    RSR = dubins_RSR(alpha, beta, d)
    if RSR is not None:
        paths.append(('RSR', RSR))

    LSR = dubins_LSR(alpha, beta, d)
    if LSR is not None:
        paths.append(('LSR', LSR))

    RSL = dubins_RSL(alpha, beta, d)
    if RSL is not None:
        paths.append(('RSL', RSL))

    RLR = dubins_RLR(alpha, beta, d)
    if RLR is not None:
        paths.append(('RLR', RLR))

    LRL = dubins_LRL(alpha, beta, d)
    if LRL is not None:
        paths.append(('LRL', LRL))

    if not paths:
        raise RuntimeError("No Dubins path found")

    # Find minimal path length sum(t + p + q)
    best_path = min(paths, key=lambda x: sum(abs(seg) for seg in x[1]))
    path_type, (t, p, q) = best_path

    lengths = [abs(t)*radius, abs(p)*radius, abs(q)*radius]

    # px, py, pyaw can be generated here if needed
    px, py, pyaw = [], [], []

    return px, py, pyaw, path_type, lengths
