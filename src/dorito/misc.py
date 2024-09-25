def clip_io_ramp(exposures: list, n_groups: int):
    new_exposures = []

    for exp in exposures:
        up_to = 45 - n_groups
        for attr in ["slopes", "variance", "ramp", "ramp_variance"]:
            exp = exp.set(attr, exp.get(attr)[:-up_to, ...])
        new_exposures.append(exp)

    return new_exposures