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

from SiEPIC._globals import Python_Env

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
            go = False
            if Python_Env == 'KLayout_GUI':
                install = pya.MessageBox.warning(
                    "Install package?", "Install package '%s' using pip? %s" % (package, request_comment),  pya.MessageBox.Yes + pya.MessageBox.No)
                if install == pya.MessageBox.Yes:
                    go = True
            else:
                # if in Script mode, install numpy
                go = True
            if go:
                # try installing using pip
                from SiEPIC.install import get_pip_main
                main = get_pip_main()
                # Try installing it. Exit code 1 if it fails
                if main(['install', package]) != 0:
                    # Try installing it with "py" in front, e.g., yaml -> pyyaml
                    if main(['install', 'py'+package]) != 0:
                        return False   
            else:
                print('Not installing %s' % package)
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

def install_dependencies():
    install_ssh()


def install_pygithub():
    pass

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
    return install('numpy')
 
def install_scipy():
    return install('scipy')

def install_imageio():
    return install('imageio')

def install_potrace():
    return install('potrace')

def install_matplotlib():
    return install('matplotlib')

def install_tidy3d():
    return install('tidy3d')

def install_urllib3():
    return install('urllib3')

def install_SiEPICLabTestParam3():
    return install('SiEPICLabTestParam3')

def install_pyqt5():
    return install('pyqt5', 'PyQt5')
