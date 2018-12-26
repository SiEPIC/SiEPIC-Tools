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
    try:
        import numpy
    except:
        try:
            import pip
            import pya
            install = pya.MessageBox.warning(
                "Install package?", "Install package 'numpy' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
            if install == pya.MessageBox.Yes:
                # try installing using pip
                from SiEPIC.install import get_pip_main
                main = get_pip_main()
                main(['install', 'numpy'])
        except ImportError:
            pass
    return True


def install_scipy():
    try:
        import scipy
    except:
        try:
            import pip
            import pya
            install = pya.MessageBox.warning(
                "Install package?", "Install package 'scipy' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
            if install == pya.MessageBox.Yes:
                # try installing using pip
                from SiEPIC.install import get_pip_main
                main = get_pip_main()
                main(['install', 'scipy'])
        except ImportError:
            pass
    return True
