import sys, os, pya

def install_dependencies():
  check_dependency('cryptography')
  check_dependency('paramiko')
  check_dependency('certifi')
  check_dependency('chardet')
  check_dependency('idna')
  check_dependency('requests')
  check_dependency('svgwrite')
  check_dependency('urllib3')
  check_dependency('numpy')
  check_dependecny('PyGithub')
  
def install_lumapi():
  pass
  
def check_dependency(module):
  try:
    check_external_python_install()
    return __import__(module)
  except ImportError:
    install = pya.QMessageBox()
    install.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel)
    install.setDefaultButton(pya.QMessageBox.Yes)
    install.setText("Error: This tool requires " + module + " in order to run.")
    install.setInformativeText("Would you like SiEPIC to install it?")
    if(pya.QMessageBox_StandardButton(install.exec_()) == pya.QMessageBox.Cancel):
      return False
    else:
      if not check_external_python_install():
        install.setText("Error: SiEPIC needs Anaconda3 for Python 3.4 to run on Windows.")
        install.setInformativeText("Please install Anaconda and create an environment for Python 3.4. For more information visit https://conda.io/docs/user-guide/tasks/manage-python.html. When finished, click continue.")
        if(pya.QMessageBox_StandardButton(install.exec_()) == pya.QMessageBox.Cancel):
          return False
        else:
          if not setup_anaconda():
            install.setStandardButtons(pya.QMessageBox.Ok)
            install.setDefaultButton(pya.QMessageBox.Ok)
            install.setText("Error: Could not install required module.")
            if(install.exec_()): return False
      if not sys.platform.startswith('win'):
        try:
          import pip
          pip.main(['install', module])
          return True
        except ImportError:
          install.setStandardButtons(pya.QMessageBox.Ok)
          install.setDefaultButton(pya.QMessageBox.Ok)
          install.setText("Error: Pip not installed. Could not install required module.")
          install.setInformativeText("")
          if(install.exec_()): return False
        except:
          print("Unexpected error:", sys.exc_info()[0])
          install.setStandardButtons(pya.QMessageBox.Ok)
          install.setDefaultButton(pya.QMessageBox.Ok)
          install.setText("Error: Could not install required module.")
          install.setInformativeText("")
          if(install.exec_()): return False
      else:
        try:
          import subprocess
          anaconda = pya.Application.instance().get_config('siepic-tools-anaconda')
          subprocess.run(anaconda + "\\Scripts\\activate.bat " + anaconda + "\\envs\\py34 && pip install " + module + " && " + anaconda + "\\Scripts\\deactivate.bat", shell=True)
          return True
        except:
          print("Unexpected error:", sys.exc_info()[0])
          install.setStandardButtons(pya.QMessageBox.Ok)
          install.setDefaultButton(pya.QMessageBox.Ok)
          install.setText("Error: Could not install required module.")
          install.setInformativeText("")
          if(install.exec_()): return False
  
def check_external_python_install():
  if not sys.platform.startswith('win'): return True
  if os.path.isdir(pya.Application.instance().get_config('siepic-tools-anaconda')):
    path_a = os.path.join(pya.Application.instance().get_config('siepic-tools-anaconda'), 'envs', 'py34', 'Lib')
    path_b = os.path.join(pya.Application.instance().get_config('siepic-tools-anaconda'), 'envs', 'py34', 'Lib', 'site-packages')
    if (path_a in sys.path) and (path_b in sys.path):
      return True
    else:
      if os.path.isdir(path_a):
        sys.path.append(path_a)
      else:
        return False
      if os.path.isdir(path_b):
        sys.path.append(path_b)
      else:
        return False
      return True
  else:
    return False

def setup_anaconda():
  anaconda = pya.Application.instance().get_config('siepic-tools-anaconda')
  if os.path.isdir(os.path.join(anaconda, 'envs', 'py34')):
    sys.path.append(os.path.join(anaconda, 'envs', 'py34', 'Lib'))
    sys.path.append(os.path.join(anaconda, 'envs', 'py34', 'Lib', 'site-packages'))
    return True
  elif os.path.isdir(os.path.join(os.environ['USERPROFILE'], 'Anaconda3', 'envs', 'py34')):
    pya.Application.instance().set_config('siepic-tools-anaconda', os.path.join(os.environ['USERPROFILE'], 'Anaconda3'))
    sys.path.append(os.path.join(pya.Application.instance().get_config('siepic-tools-anaconda'), 'envs', 'py34', 'Lib'))
    sys.path.append(os.path.join(pya.Application.instance().get_config('siepic-tools-anaconda'), 'envs', 'py34', 'Lib', 'site-packages'))
    return True
  else:
    install = pya.QMessageBox()
    install.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel)
    install.setDefaultButton(pya.QMessageBox.Yes)
    install.setText("Error: Anaconda was not found.")
    install.setInformativeText("Would you like to specify Anaconda's install location manually?")
    if(pya.QMessageBox_StandardButton(install.exec_()) == pya.QMessageBox.Cancel):
      return False
    else:
      anaconda = pya.QFileDialog().getExistingDirectory()
      if os.path.isdir(os.path.join(anaconda, 'envs', 'py34')):
        pya.Application.instance().set_config('siepic-tools-anaconda', anaconda)
        sys.path.append(os.path.join(pya.Application.instance().get_config('siepic-tools-anaconda'), 'envs', 'py34', 'Lib'))
        sys.path.append(os.path.join(pya.Application.instance().get_config('siepic-tools-anaconda'), 'envs', 'py34', 'Lib', 'site-packages'))
        return True
      else:
        install.setStandardButtons(pya.QMessageBox.Ok)
        install.setDefaultButton(pya.QMessageBox.Ok)
        install.setText("Error: Python 3.4 environment not found in this directory.")
        if(install.exec_()): return False