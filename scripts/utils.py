import os


def create_dir(directory):
    if not os.path.isdir(directory):
        _path = os.path.abspath(directory).split('\\')
        for i in range(1, len(_path) + 1):
            current_dir = "//".join(_path[:i])
            if not os.path.isdir(current_dir):
                os.mkdir(current_dir)