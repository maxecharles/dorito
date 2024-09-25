def truncate_files(files, ngroups, top_group):
    """
    Truncate the ramp of files to only have ngroups.
    """
    # TODO have it read top group from the files
    up_to = top_group - ngroups
    
    for file in files:
        # files are mutable, so they will change in place
        for attr in ["RAMP", "SLOPE", "RAMP_ERR", "SLOPE_ERR", "RAMP_SUP", "SLOPE_SUP"]:
            file[attr].data = file[attr].data[:-up_to, ...]