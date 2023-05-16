import os

DATA_PATH = "../data"


def data_path(path):
    return os.path.join(DATA_PATH, path)


def read_proxies(path, user=None, password=None):
    """Read proxies from txt list

    Either edit user and pass directly here, or provide upon function usage
    """

    assert user is not None and password is not None

    with open(path, "r") as handle:
        proxylist = [prox.strip() for prox in handle]

    return [f"http://{user}:{password}@{proxy}" for proxy in proxylist]
