def install_ssh():
  pass
  #import pip, os, pya
  #path = os.path.join(os.path.dirname(os.path.realpath(pya.Application.instance().klayout_path()[-1])), "lib", "python")
  #pip.main(['install', 'cryptography', '--no-index', '--find-links='+path])
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
  