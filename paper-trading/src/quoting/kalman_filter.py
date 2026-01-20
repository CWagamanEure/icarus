from .models import MicroFeatures, NoiseModelParams, KFState

def obs_var_Rt(features: MicroFeatures, params: NoiseModelParams) -> float:
    spread = max(features.spread, 1e-6)
    depth = max(features.depth_bid + features.depth_ask, 1e-6)
    imb = features.imbalance
    R = (
        params.a0
        + params.a_spread * (spread ** 2)
        + params.a_depth * (1.0 / depth)
        + params.a_imb * (imb ** 2)
    )
    return max(R, 1e-12)


def kf_step_random_walk(state: KFState, y: float, R: float, Q:float) -> KFState:
    """
    Given current KF state and new info, computes new state
    """

    # predict
    x_pred = state.x
    P_pred = state.P + Q

    #update
    K = P_pred / (P_pred + R)
    innov = y - x_pred
    x_new = x_pred + K * innov
    P_new = (1.0 - K) * P_pred

    return (KFState(x=x_new, P=P_new))


