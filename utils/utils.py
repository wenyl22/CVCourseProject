import os
def mkdirs(paths):
    """create empty directories if they don't exist
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
