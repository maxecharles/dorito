def truncate_files(files, ngroups, top_group):
    """
    Truncate the ramp of files to only have ngroups.
    """
    # TODO have it read top group from the files
    # TODO do one_on_fs here
    up_to = top_group - ngroups

    for file in files:
        # files are mutable, so they will change in place
        for attr in ["RAMP", "SLOPE", "RAMP_ERR", "SLOPE_ERR", "RAMP_SUP", "SLOPE_SUP"]:
            file[attr].data = file[attr].data[:-up_to, ...]


def combine_param_dicts(cal_params, sci_params):
    """
    Combining the calibration and science parameter dictionaries.
    """
    params = {**cal_params}

    # to avoid doubling up
    for key in sci_params:
        if key in cal_params:
            params[key] = cal_params[key] | sci_params[key]
        else:
            params[key] = sci_params[key]

    return params
