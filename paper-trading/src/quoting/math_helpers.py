import math

EPS = 1e-9


def clamp(x:float, lo:float, hi:float) -> float:
    """
    clamp between prob range
    """
    return max(lo, min(hi, x))

def logit(p: float) -> float:
    """
    prob to logit
    """
    p = clamp(p, EPS, 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
