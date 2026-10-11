import numpy as np

# Extended Kalman Filter
#
# State vector (7x1 column, float32 like OpenCV):
#   [x, y, z, theta, vx, vy, vz]
#   x, y, z     - position
#   theta       - roll angle (rad)
#   vx, vy, vz  - velocity
#
# Measurement from OpenCV (4x1 column): [x, y, z, theta]


# ---- Process model: f and its Jacobian F ----

def f(state_prev, dt):
    """State transition f(x_{k-1}): constant velocity, roll held constant.
        x_k = x + dt * vx
        y_k = y + dt * vy
        z_k = z + dt * vz
        theta_k = theta
        vx, vy, vz stay the same
    """
    x, y, z, theta, vx, vy, vz = np.asarray(state_prev, dtype=np.float32).ravel()
    return np.array([[x + dt * vx],
                     [y + dt * vy],
                     [z + dt * vz],
                     [theta],
                     [vx],
                     [vy],
                     [vz]], dtype=np.float32)

def F_jacobian(state_prev, dt):
    """Jacobian of f, d f / d state (the EKF uses this to propagate covariance).
    f is linear here, so this is just the transition matrix.
    Must be re-derived whenever f changes."""
    F = np.eye(7, dtype=np.float32)
    F[0, 4] = dt  # dx/dvx
    F[1, 5] = dt  # dy/dvy
    F[2, 6] = dt  # dz/dvz
    return F

# ---- Measurement model: h and its Jacobian H ----

def h(state):
    """Measurement function h(x): what OpenCV should report for this state.
    OpenCV gives x, y, z, theta, which are the first 4 state values."""
    return np.asarray(state, dtype=np.float32).reshape(7, 1)[:4]

def H_jacobian(state):
    """Jacobian of h, d h / d state."""
    return np.hstack([np.eye(4), np.zeros((4, 3))]).astype(np.float32)

# ---- EKF ----

class ExtendedKalmanFilter:
    """EKF with the same predict()/correct() interface as cv2.KalmanFilter."""

    def __init__(self, Q, R, initial_state, P0):
        self.Q = np.asarray(Q, dtype=np.float32)    # process noise covariance
        self.R = np.asarray(R, dtype=np.float32)    # measurement noise covariance
        self.x = np.asarray(initial_state, dtype=np.float32).reshape(7, 1)
        self.P = np.asarray(P0, dtype=np.float32)   # state covariance

    def predict(self, dt):
        """Move the state forward one step using f."""
        F = F_jacobian(self.x, dt)
        self.x = f(self.x, dt)
        self.P = F @ self.P @ F.T + self.Q
        return self.x

    def correct(self, measurement):
        """Adjust the state using an OpenCV measurement [x, y, z, theta]."""
        z = np.asarray(measurement, dtype=np.float32).reshape(4, 1)
        H = H_jacobian(self.x)
        y = z - h(self.x)                         # measured minus predicted
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)       # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(7, dtype=np.float32) - K @ H) @ self.P
        return self.x

#Just a test
if __name__ == "__main__":
    dt = 1 / 30  # e.g. a 30 fps camera
    rng = np.random.default_rng(0)

    # Simulated "true" object: starts at (0, 0, 0.3), roll 0.5, velocity (1, 0.5, 0)
    true_state = np.array([0.0, 0.0, 0.3, 0.5, 1.0, 0.5, 0.0], dtype=np.float32)
    meas_std = np.array([0.02, 0.02, 0.02, 0.05])  # OpenCV noise for x, y, z, theta

    # Start the filter at the first measurement, with unknown velocity (0)
    first_meas = true_state[:4] + rng.normal(0, meas_std)
    ekf = ExtendedKalmanFilter(
        Q=np.diag([1e-5, 1e-5, 1e-5, 1e-4, 1e-3, 1e-3, 1e-3]),
        R=np.diag(meas_std**2),
        initial_state=[*first_meas, 0, 0, 0],
        P0=np.diag([*meas_std**2, 1.0, 1.0, 1.0]),
    )

    for k in range(1, 31):
        true_state = f(true_state, dt).ravel()
        measurement = true_state[:4] + rng.normal(0, meas_std)  # stands in for OpenCV

        ekf.predict(dt)
        state = ekf.correct(measurement)

        if k % 5 == 0:
            x, y, z, theta, vx, vy, vz = state.ravel()
            print(f"k={k:2d}  pos=({x:.2f}, {y:.2f}, {z:.2f})  theta={theta:.2f}  "
                  f"vel=({vx:.2f}, {vy:.2f}, {vz:.2f})")
