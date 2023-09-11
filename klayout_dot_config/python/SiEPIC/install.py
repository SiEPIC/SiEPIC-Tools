'''

Installing Python packages inside KLayout is sometimes possibles, and SiEPIC tools requires some packages
that don't come preinstalled

The preferred function is "install" which is simple to use

Usage:

# Required packages
from SiEPIC.install import install
if not install('scipy', requested_by='Contra Directional Coupler design'):
  pya.MessageBox.warning(
  "Missing package", "The simulator does not function without the package 'scipy'.",  pya.MessageBox.Ok)    


by Lukas Chrostowski, 2023

'''


def install(package, module=None, requested_by=None):
    '''Install the Python package, and import
    package: the name you need to pass to pip import
    module: some packages have a different name, e.g., 
        import nextcloud, for pip install nextcloud-api-wrapper
        import yaml, for pip install pyyaml

    '''
    if requested_by:
        request_comment = '[required for %s]' % requested_by
    else:
        request_comment = ''

    import importlib
    try:
        if module:
            importlib.import_module(module)
        else:
            importlib.import_module(package)
    except ImportError:
        try:
            import pip
            import pya
            install = pya.MessageBox.warning(
                "Install package?", "Install package '%s' using pip? %s" % (package, request_comment),  pya.MessageBox.Yes + pya.MessageBox.No)
            if install == pya.MessageBox.Yes:
                # try installing using pip
                from SiEPIC.install import get_pip_main
                main = get_pip_main()
                # Try installing it. Exit code 1 if it fails
                if main(['install', package]) != 0:
                    # Try installing it with "py" in front, e.g., yaml -> pyyaml
                    if main(['install', 'py'+package]) != 0:
                        return False   
            else:
                print('Not installing %s' %package)
                return False
        except ImportError:
            return False
            
    if module:
        globals()[package] = importlib.import_module(module)
    else:
        globals()[package] = importlib.import_module(package)
    return globals()[package]


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
    if hasattr(pip, 'main'):
        return pip.main
    else:
        from pip import main
        return main
        return pip._internal.main.main


def install_numpy():
    try:
        import numpy
    except:
        try:
            import pip
            try:
                import pya
                install = pya.MessageBox.warning(
                    "Install package?", "Install package 'numpy' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
                if install == pya.MessageBox.Yes:
                    # try installing using pip
                    from SiEPIC.install import get_pip_main
                    main = get_pip_main()
                    main(['install', 'numpy'])
            except ImportError:
                install = pya.MessageBox.warning(
                    "Install using pip failed", "Error importing 'pip' to install package 'numpy'",  pya.MessageBox.Yes + pya.MessageBox.No)
                return False
        except ImportError:
            return False
    return True
    import numpy


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
            return False
    return True
    import scipy

def install_imageio():

    try:
        import imageio
    except:
        try:
            import pip
            import pya
            install = pya.MessageBox.warning(
                "Install package?", "Install package 'imageio' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
            if install == pya.MessageBox.Yes:
                # try installing using pip
                from SiEPIC.install import get_pip_main
                main = get_pip_main()
                main(['install', 'imageio'])
        except ImportError:
            return False
    return True

def install_potrace():

    try:
        import potrace
    except:
        try:
            import pip
            import pya
            install = pya.MessageBox.warning(
                "Install package?", "Install package 'potrace' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
            if install == pya.MessageBox.Yes:
                # try installing using pip
                from SiEPIC.install import get_pip_main
                main = get_pip_main()
                main(['install', 'potrace'])
        except ImportError:
            return False
    return True


def install_matplotlib():
    try:
        import matplotlib
    except:
        try:
            import pip
            import pya
            install = pya.MessageBox.warning(
                "Install package?", "Install package 'matplotlib' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
            if install == pya.MessageBox.Yes:
                # try installing using pip
                from SiEPIC.install import get_pip_main
                main = get_pip_main()
                main(['install', 'matplotlib'])
        except ImportError:
            return False
    return True
    import matplotlib

def install_tidy3d():
    try:
        import tidy3d
    except:
        try:
            import pip
            import pya
            install = pya.MessageBox.warning(
                "Install package?", "Install package 'tidy3d' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
            if install == pya.MessageBox.Yes:
                # try installing using pip
                from SiEPIC.install import get_pip_main
                main = get_pip_main()
                main(['install', 'tidy3d'])
        except ImportError:
            return False
    return True
    import tidy3d

def install_urllib3():
    try:
        import urllib3
    except:
        try:
            import pip
            import pya
            install = pya.MessageBox.warning(
                "Install package?", "Install package 'urllib3' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
            if install == pya.MessageBox.Yes:
                # try installing using pip
                from SiEPIC.install import get_pip_main
                main = get_pip_main()
                main(['install', 'urllib3'])
        except ImportError:
            return False
    return True

def install_SiEPICLabTestParam3():
    try:
        import SiEPICLabTestParam
    except:
        try:
            import pip
            import pya
            install = pya.MessageBox.warning(
                "Install package?", "Install package 'SiEPICLabTestParam3' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
            if install == pya.MessageBox.Yes:
                # try installing using pip
                from SiEPIC.install import get_pip_main
                main = get_pip_main()
                main(['install', 'SiEPICLabTestParam3'])
        except ImportError:
            return False
    return True
    import SiEPICLabTestParam



def install_pyqt5():
    try:
        import PyQt5
    except:
        try:
            import pip
            import pya
            install = pya.MessageBox.warning(
                "Install package?", "Install package 'pyqt5' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
            if install == pya.MessageBox.Yes:
                # try installing using pip
                from SiEPIC.install import get_pip_main
                main = get_pip_main()
                main(['install', 'pyqt5'])
        except ImportError:
            return False
    return True
    import PyQt5


