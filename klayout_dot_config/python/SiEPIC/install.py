def install_ssh():
    pass
    #import pip, os, pya
    #pip.main(['install', 'cryptography'])
    #pip.main(['install', 'paramiko'])


def install_dependencies():
    install_ssh()


def install_pygithub():
    pass

    # OSX:
    # easy_install PyGithub
    # pip for python: https://stackoverflow.com/questions/17271319/how-do-i-install-pip-on-macos-or-os-x
    #   sudo easy_install pip
    #   sudo pip install PyGithub


def install_lumapi():
    pass


def get_pip_main():
    import pip
    # check if pip version is new:
    if int(pip.__version__.split('.')[0]) > 9:
        from pip._internal import main
    else:
        from pip import main
    return main


def install_numpy():
    main = get_pip_main()
    main(['install', 'numpy'])
    return True


def install_scipy():
    main = get_pip_main()
    main(['install', 'scipy'])
    return True
