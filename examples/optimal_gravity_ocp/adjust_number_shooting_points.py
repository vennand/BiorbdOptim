# --- Adjust number of shooting points --- #
def adjust_number_shooting_points(number_shooting_points, frames):
    list_adjusted_number_shooting_points = []
    for frame_num in range(1, (abs(frames.stop - frames.start) - 1) // abs(frames.step) + 1):
        list_adjusted_number_shooting_points.append((abs(frames.stop - frames.start) - 1) // frame_num + 1)
    diff_shooting_points = [abs(number_shooting_points - point) for point in list_adjusted_number_shooting_points]
    step_size = diff_shooting_points.index(min(diff_shooting_points)) + 1
    adjusted_number_shooting_points = ((abs(frames.stop - frames.start) - 1) // step_size + 1) - 1

    return adjusted_number_shooting_points, step_size
