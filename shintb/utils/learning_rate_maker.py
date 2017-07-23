def learning_rate_maker(global_step, learning_rate_list):  # global_step : tf.float32
    if global_step < 4000 :
        return learning_rate_list[0]
    elif global_step < 180000 :
        return learning_rate_list[1]
    elif global_step < 240000 :
        return learning_rate_list[2]
    else :
        return learning_rate_list[3]