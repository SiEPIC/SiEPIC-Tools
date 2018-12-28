def check_numpy():
    ''' Check if numpy is present
    '''
    try:
        import numpy  # noqa
    except ImportError:
        return False
    else:
        return True
