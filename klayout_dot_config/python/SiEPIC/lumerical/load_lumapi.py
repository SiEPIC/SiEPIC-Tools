
def load_lumapi():
  import os, platform, sys, inspect
  if platform.system() == 'Darwin':
    path_fdtd = "/Applications/Lumerical/FDTD Solutions/FDTD Solutions.app/Contents/API/Python"
    if os.path.exists(path_fdtd):
      path = path_fdtd
    path_intc = "/Applications/Lumerical/INTERCONNECT/INTERCONNECT.app/Contents/API/Python"
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
    
  CWD = os.path.dirname(os.path.abspath(__file__))
  
  
  if platform.system() == 'Darwin':
    # Check if any Lumerical tools are installed
    if os.path.exists(path):
      ##################################################################
      # Configure OSX Path to include Lumerical tools: 
            
      # Copy the launch control file into user's Library folder
      # execute launctl to register the new paths
      import os, fnmatch, commands
      
      siepic_tools_lumerical_folder = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
  
      filename = siepic_tools_lumerical_folder + '/SiEPIC_Tools_Lumerical_KLayout_environment.plist'
      if not os.path.exists(filename):
        raise Exception ('Missing file: %s' % filename)
  
      # Check if Paths are correctly set, and KLayout Python sees them
      a,b=commands.getstatusoutput('echo $SiEPIC_Tools_Lumerical_KLayout_environment')
      if b=='':
        # Not yet installed... copy files, install
        cmd1='launchctl unload  %s' % filename
        a,b=commands.getstatusoutput(cmd1)
        if a != 0:
          raise Exception ('Error calling: %s, %s' % (cmd1, b) )
        cmd1='launchctl load  %s' % filename
        a,b=commands.getstatusoutput(cmd1)
        if a != 0 or b !='':
          raise Exception ('Error calling: %s, %s' % (cmd1, b) )
        cmd1='killall Dock'
        a,b=commands.getstatusoutput(cmd1)
        if a != 0 or b !='':
          raise Exception ('Error calling: %s, %s' % (cmd1, b) )
  
        # Check if Paths are correctly set, and KLayout Python sees them
        a,b=commands.getstatusoutput('echo $SiEPIC_Tools_Lumerical_KLayout_environment')
        if b=='':
          # Not loaded    
          raise Exception ('The System paths have been updated. Please restart KLayout to use Lumerical tools.')
  
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
  
      # Fix for Lumerical Python OSX API:
      if not path in sys.path:
        sys.path.append(path)
  #    os.chdir(path) 
      lumapi_osx_fix = siepic_tools_lumerical_folder + '/lumapi_osx_fix.bash'
      if not os.path.exists(lumapi_osx_fix):
        print (commands.getstatusoutput('chmod a+x %s' % lumapi_osx_fix ))
        print (commands.getstatusoutput('/Applications/Utilities/Terminal.app/Contents/MacOS/Terminal %s' %lumapi_osx_fix ))
      
  # Windows
  elif platform.system() == 'Windows': 
    if os.path.exists(path):
      if not path in sys.path:
        sys.path.append(path) # windows
      os.chdir(path) 

  # for all operating systems:
  from .. import _globals
  if not _globals.LUMAPI:
    print('import lumapi')
    import lumapi
    _globals.LUMAPI = lumapi    
#    _globals.INTC = lumapi.open('interconnect')
#    _globals.FDTD = lumapi.open('fdtd')
  
  os.chdir(CWD)
  
load_lumapi()
