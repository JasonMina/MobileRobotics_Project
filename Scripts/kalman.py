import numpy as np

d = 98.0 # mm
N = 6

Q = np.eye(N, dtype = float);
Q[0, 0] = 0.1  #position x
Q[1, 1] = 6.15 #speed x
Q[2, 2] = 0.1  #position y
Q[3, 3] = 6.15 #speed y
Q[4, 4] = 0.01  #position theta
Q[5, 5] = 0.13  #speed theta

thspeed2speed = 0.341

x_est_prev = np.zeros(6)
P_est_prev = np.eye(6, dtype = float)

def init_kalman(x_0, P_0):
    global x_est_prev, P_est_prev

    x_est_prev = x_0
    P_est_prev = P_0

def kalman_filter(Ts, x_speed, y_speed, theta_speed, is_vision, x_m, y_m, theta_m, x_est_prev, P_est_prev):

    """
    Estimates the current state using input sensor data and the previous state
    
    param x_speed    : mesured speed in x (mm/s)
    param y_speed    : mesured speed in y (mm/s)
    param theta_speed: mesured rotation speed (rad/s)
    param is_vision  : 0 if the vision is blocked, 1 if the vision is here
    param x_m        : mesured position in x (mm)
    param y_m        : mesured position in y (mm)
    param theta_m    : mesured angle (rad)
    param x_est_prev : previous state a posteriori estimation
    [[      x_pos],
     [    x_speed],
     [      y_pos],
     [    y_speed],
     [      theta],
     [theta_speed]]
     
    param P_est_prev : previous state a posteriori covariance
    
    return x_est: new a posteriori state estimation
    return P_est: new a posteriori state covariance
    """
    
    A = np.array([[  1, Ts,  0,  0,  0,  0],
                  [  0,  1,  0,  0,  0,  0],
                  [  0,  0,  1, Ts,  0,  0],
                  [  0,  0,  0,  1,  0,  0],
                  [  0,  0,  0,  0,  1, Ts],
                  [  0,  0,  0,  0,  0,  1]])

    ## Prediciton through the a priori estimate
    # Predict mean of the state
    x_est_a_priori = np.dot(A, x_est_prev);
    
    # Predict covariance of the state
    P_est_a_priori = np.dot(A, np.dot(P_est_prev, A.T))
    P_est_a_priori = P_est_a_priori + Q
    
    
    ## Estimation         
    # y, C, and R for a posteriori estimate, depending on transition
    if (is_vision):
        y = np.array([[x_m],
                      [x_speed],
                      [y_m],
                      [y_speed],
                      [theta_m],
                      [theta_speed]])
        
        H = np.eye(6, dtype = float)
        
        R = np.eye(6, dtype = float)
        R[0, 0] = 0.1  #position x
        R[1, 1] = 6.15 #speed x
        R[2, 2] = 0.1  #position y
        R[3, 3] = 6.15 #speed y
        R[4, 4] = 0.01  #position theta
        R[5, 5] = 0.13  #speed theta
        
    else:
        # no transition, use only the speed
        y = np.array([[x_speed],
                      [y_speed],
                      [theta_speed]])
        
        H = np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 1]])
        
        R = np.eye(3, dtype = float)
        R[0, 0] = 6.15 #speed x
        R[1, 1] = 6.15 #speed y
        R[2, 2] = 0.13  #speed theta

        
    ## estimator error
    i = y.flatten() - np.dot(H, x_est_a_priori) #measurement residual
    
    ## Calculation of matrix K
    S = np.dot(H, np.dot(P_est_a_priori, H.T)) + R # measurement prediction covariance
    K = np.dot(P_est_a_priori, np.dot(H.T, np.linalg.inv(S))) # Kalman gain
    
    # a posteriori estimate
    x_est = x_est_a_priori + np.dot(K,i)
    P_est = P_est_a_priori - np.dot(K,np.dot(H, P_est_a_priori))

    return x_est, P_est

# Updates the position by calling the Kalman filter with or without camera based values
def update_position(Ts, pos_x_m, pos_y_m, theta_m, vision, motor_left_speed, motor_right_speed):
    global x_est_prev, P_est_prev, end, start

    speed_x = thspeed2speed*(motor_left_speed+motor_right_speed)/2.0 * np.cos(x_est_prev[4])
    speed_y = thspeed2speed*(motor_left_speed+motor_right_speed)/2.0 * np.sin(x_est_prev[4])
    speed_w = thspeed2speed*(motor_right_speed-motor_left_speed)/(d)

    x_est, P_est = kalman_filter(Ts, speed_x, speed_y, speed_w, vision, pos_x_m, pos_y_m, theta_m, x_est_prev, P_est_prev)

    x_est_prev = x_est
    P_est_prev = P_est

    return x_est

def err_angle(x, y, theta, x1, y1):
    err = - theta + np.arctan2((y1-y), (x1 - x))

    if abs(err) <= np.pi:
        return err
    elif err > np.pi:
        return 2*np.pi - err
    elif err < -np.pi:
        return 2*np.pi + err
