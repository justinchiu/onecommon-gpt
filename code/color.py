# dots are x, y, size, color
 
def is_black(dot):
    return dot[-1] > 0.4

def is_light(dot):
    # colors are in [-1,1]
    return dot[-1] < -0.4

def is_grey(dot):
    return not is_black(dot) and not is_light(dot)
