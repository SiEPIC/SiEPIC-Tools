import pya

def load_lumapi(verbose=False):
  import pya
  if verbose:
    print("SiEPIC.lumerical.load_lumapi")


  import sys


  if 'numpy' not in sys.modules:
      try:
          import pip
      except ImportError:
          pass
      if 'numpy' in sys.modules:
          import pya
          install = pya.MessageBox.warning(
              "Install package?", "Install package 'numpy' using pip?",  pya.MessageBox.Yes + pya.MessageBox.No)
          if install == pya.MessageBox.Yes:
              # try installing using pip
              from SiEPIC.install import get_pip_main
              main = get_pip_main()
              main(['install', 'numpy'])
  

  try:
    import numpy
  except:
    print('Missing numpy. Cannot load Lumerical Python integration')
    warning = pya.QMessageBox()
    warning.setStandardButtons(pya.QMessageBox.Cancel)
    warning.setText("Missing Python module numpy.  \nCannot load Lumerical Python integration. ") 
    warning.setInformativeText("Some SiEPIC-Tools Lumerical functionality will not be available.\nPlease install numpy.  For Windows users, install the Package Windows_Python_packages_for_KLayout.")
    pya.QMessageBox_StandardButton(warning.exec_())
    return

  import os, platform, sys, inspect

  # Load the Lumerical software location from KLayout configuration
  path = pya.Application.instance().get_config('siepic_tools_Lumerical_Python_folder')

  # if it isn't defined, start with Lumerical's defaults
  if not path:
    if platform.system() == 'Darwin':
      path_fdtd = "/Applications/Lumerical/FDTD Solutions/FDTD Solutions.app/Contents/API/Python"
      if os.path.exists(path_fdtd):
        path = path_fdtd
      path_intc = "/Applications/Lumerical/INTERCONNECT/INTERCONNECT.app/Contents/API/Python"
      if os.path.exists(path_intc):
        path = path_intc
    elif platform.system() == 'Linux':
      path_fdtd = "/opt/lumerical/fdtd/api/python"
      if os.path.exists(path_fdtd):
        path = path_fdtd
      path_intc = "/opt/lumerical/interconnect/api/python"
      if os.path.exists(path_intc):
        path = path_intc
    elif platform.system() == 'Windows': 
      path_fdtd = "C:\\Program Files\\Lumerical\\FDTD Solutions\\api\\python"
      if os.path.exists(path_fdtd):
        path = path_fdtd
      path_intc = "C:\\Program Files\\Lumerical\\INTERCONNECT\\api\\python"
      if os.path.exists(path_intc):
        path = path_intc
    else:
      print('Not a supported OS')
      return

  # if it is still not found, ask the user
  if not os.path.exists(path):
    print('SiEPIC.lumerical.load_api: Lumerical software not found')
    question = pya.QMessageBox()
    question.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.No)
    question.setDefaultButton(pya.QMessageBox.Yes)
    question.setText("Lumerical software not found. \nDo you wish to locate the software?")
    if(pya.QMessageBox_StandardButton(question.exec_()) == pya.QMessageBox.Yes):
      p = pya.QFileDialog()
      p.setFileMode(pya.QFileDialog.DirectoryOnly)
      p.exec_()
      path = p.directory().path
      if verbose:
        print(path)
    else:
      return    
      
  # check if we have the correct path, containing lumapi.py
  if not os.path.exists(os.path.join(path,'lumapi.py')):
    # check sub-folders for lumapi.py
    import fnmatch
    dir_path = path
    search_str = 'lumapi.py'
    matches = []
    for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
        for filename in fnmatch.filter(filenames, search_str):
            matches.append(root)
    if matches:
      if verbose:
        print(matches)
      path = matches[0]
      
    if not os.path.exists(os.path.join(path,'lumapi.py')):
      print('SiEPIC.lumerical.load_api: Lumerical lumapi.py not found')
      warning = pya.QMessageBox()
      warning.setStandardButtons(pya.QMessageBox.Cancel)
      warning.setText("Lumerical's lumapi.py not found.")
      warning.setInformativeText("Some SiEPIC-Tools Lumerical functionality will not be available.")
      pya.QMessageBox_StandardButton(warning.exec_())
      return
    
  # Save the Lumerical software location to the KLayout configuration
  pya.Application.instance().set_config('siepic_tools_Lumerical_Python_folder', path)

      
  CWD = os.path.dirname(os.path.abspath(__file__))
  
  
  if platform.system() == 'Darwin':
    # Check if any Lumerical tools are installed
      ##################################################################
      # Configure OSX Path to include Lumerical tools: 
            
      # Copy the launch control file into user's Library folder
      # execute launctl to register the new paths
      import os, fnmatch
      siepic_tools_lumerical_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

      os.environ['PATH'] += ':/Applications/Lumerical/FDTD Solutions/FDTD Solutions.app/Contents/MacOS' 
      os.environ['PATH'] += ':/Applications/Lumerical/INTERCONNECT/INTERCONNECT.app/Contents/MacOS' 
      os.environ['PATH'] += ':/Applications/Lumerical/INTERCONNECT/INTERCONNECT.app/Contents/API/Python'
      os.environ['PATH'] += ':/Applications/Lumerical/INTERCONNECT/INTERCONNECT.app/Contents/API/Matlab'


      # Also add path for use in the Terminal
      home = os.path.expanduser("~")
      if not os.path.exists(home + "/.bash_profile"):
        text_bash =  '\n'
        text_bash += '# Setting PATH for Lumerical API\n'
        text_bash += 'export PATH=/Applications/Lumerical/FDTD\ Solutions/FDTD\ Solutions.app/Contents/MacOS:$PATH\n'
        text_bash += 'export PATH=/Applications/Lumerical/MODE\ Solutions/MODE\ Solutions.app/Contents/MacOS:$PATH\n'
        text_bash += 'export PATH=/Applications/Lumerical/DEVICE/DEVICE.app/Contents/MacOS:$PATH\n'
        text_bash += 'export PATH=/Applications/Lumerical/INTERCONNECT/INTERCONNECT.app/Contents/MacOS:$PATH\n'
        text_bash +=  '\n'
        file = open(home + "/.bash_profile", 'w')
        file.write (text_bash)
        file.close()

      if not path in sys.path:
        sys.path.append(path)
      # Fix for Lumerical Python OSX API, for < March 5 2018 versions:
      if not os.path.exists(os.path.join(path, 'libinterop-api.1.dylib')):
        lumapi_osx_fix = siepic_tools_lumerical_folder + '/lumapi_osx_fix.bash'
        lumapi_osx_fix_lib = path + '/libinterop-api.so.1'
        if not os.path.exists(lumapi_osx_fix_lib):
          warning = pya.QMessageBox()
          warning.setStandardButtons(pya.QMessageBox.Ok)
          warning.setText("We need to do a fix in the Lumerical software folder for Python integration. \nPlease note that for this to work, we assume that Lumerical INTERCONNECT is installed in the default path: /Applications/Lumerical/INTERCONNECT/\nPlease enter the following in a Terminal.App window, and enter your root password when prompted. Ok to continue when done.")
          warning.setInformativeText("source %s"%lumapi_osx_fix)
          pya.QMessageBox_StandardButton(warning.exec_())
          if not os.path.exists(lumapi_osx_fix_lib):
            import sys
            if int(sys.version[0]) > 2:
              import subprocess
              subprocess.Popen(['/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal', '-run', lumapi_osx_fix])          
            else:
              import commands
              print (commands.getstatusoutput('/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal %s' %lumapi_osx_fix ))
      
  # Windows
  elif platform.system() == 'Windows': 
    if os.path.exists(path):
      if not path in sys.path:
        sys.path.append(path) # windows
      os.chdir(path) 
  # Linux    
  elif platform.system() == 'Linux': 
    if os.path.exists(path):
      if not path in sys.path:
        sys.path.append(path) # windows
      os.chdir(path) 

  # for all operating systems:
  from .. import _globals
  if not _globals.LUMAPI:
    try:
      import lumapi
      _globals.LUMAPI = lumapi    
    except:
      print('import lumapi failed')
      return

  print('import lumapi success, %s' % _globals.LUMAPI )
#    _globals.INTC = lumapi.open('interconnect')
#    _globals.FDTD = lumapi.open('fdtd')
  
  os.chdir(CWD)
  
load_lumapi(verbose=True)
