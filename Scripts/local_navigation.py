
state = 0
gain = 500
avoidThresh = 2200
obstSpeedGain = [5, 3, -7, -4, -6]

mean_speed = 100


def local_nav(correction_speed_r, correction_speed_l, theta_err, prox_sensor):
    global gain, obstSpeedGain, margin, state, avoidThresh, mean_speed
    
    commande_speed_l = 100
    commande_speed_r = 100

    # If sense something, goes to avoidance state
    if (prox_sensor[0] >= avoidThresh
        or prox_sensor[1] >= avoidThresh
        or prox_sensor[2] >= avoidThresh
        or prox_sensor[3] >= avoidThresh
        or prox_sensor[4] >= avoidThresh):
        state = 0

    # Avoidance state
    if state == 0:
        for i in range(5):
            commande_speed_l += prox_sensor[i] * obstSpeedGain[i] // gain
            commande_speed_r += prox_sensor[i] * obstSpeedGain[4 - i] // gain


        # If doesn't sense anything, goes to moving state
        if (prox_sensor[0] < avoidThresh
            and prox_sensor[1] < avoidThresh
            and prox_sensor[2] < avoidThresh
            and prox_sensor[3] < avoidThresh
            and prox_sensor[4] < avoidThresh):
            state = 1
    
    # Go straight to the goal
    elif state == 1:
        commande_speed_l = correction_speed_l + mean_speed
        commande_speed_r = correction_speed_r + mean_speed

    return commande_speed_l, commande_speed_r