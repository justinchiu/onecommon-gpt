# dots are x, y, size, color

# -1 is darkest
def is_dark(dot):
    #return dot[-1] < -0.4
    return dot[-1] < -0.3

# 1 is lightest
def is_light(dot):
    # colors are in [-1,1]
    #return dot[-1] > 0.4
    return dot[-1] > 0.3

def is_grey(dot):
    return True
    return not is_dark(dot) and not is_light(dot)
