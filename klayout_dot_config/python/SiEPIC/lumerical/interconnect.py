'''
################################################################################
#
#  SiEPIC-Tools
#  
################################################################################

Circuit simulations using Lumerical INTERCONNECT and a Compact Model Library

- run_INTC: run INTERCONNECT using Python integration
- Setup_Lumerical_KLayoutPython_integration
    Configure PATH env, import lumapi, run interconnect, 
    Install technology CML, read CML elements
- circuit_simulation: netlist extract and run simulation
- circuit_simulation_update_netlist: update netlist and run simulation
- circuit_simulation_monte_carlo: perform many simulations
- component_simulation: single component simulation

usage:
 import SiEPIC.lumerical.interconnect:


################################################################################
'''



import pya

def run_INTC():
  import lumapi_osx as lumapi
  from .. import _globals
  _globals.INTC  # Python Lumerical INTERCONNECT integration handle
  
  if not _globals.INTC: # Not running, start a new session
    _globals.INTC = lumapi.open('interconnect')
    print(_globals.INTC)
  else: # found open INTC session
    try:
      lumapi.evalScript(_globals.INTC, "?'KLayout integration test.';")
    except: # but can't communicate with INTC; perhaps it was closed by the user
      INTC = lumapi.open('interconnect') # run again.
  try: # check again
    lumapi.evalScript(_globals.INTC, "?'KLayout integration test.';")
  except:
    raise Exception ("Can't run Lumerical INTERCONNECT. Unknown error.")

def Setup_Lumerical_KLayoutPython_integration():
  import sys, os, string
  if sys.platform.startswith('darwin'):

    if string.find(sys.version,"2.7.") > -1:
      import commands
    else:
      raise Exception ('Unknown Python version: %s' % file_name)
  
    ##################################################################
    # Configure OSX Path to include Lumerical tools: 
          
    # Copy the launch control file into user's Library folder
    # execute launctl to register the new paths
    import os, fnmatch
    dir_path = pya.Application.instance().application_data_path()
    file_name = 'SiEPIC_Tools_Lumerical_KLayout_environment.plist'
    matches = []
    for root, dirnames, filenames in os.walk(dir_path, followlinks=True):
        for filename in fnmatch.filter(filenames, file_name):
            matches.append(os.path.join(root, filename))
    
    if not matches[0]:
      raise Exception ('Missing file: %s' % file_name)

    # Check if Paths are correctly set, and KLayout Python sees them
    a,b=commands.getstatusoutput('echo $SiEPIC_Tools_Lumerical_KLayout_environment')
    if b=='':
      # Not yet installed... copy files, install
      cmd1='launchctl unload  %s' % matches[0]
      a,b=commands.getstatusoutput(cmd1)
      if a != 0:
        raise Exception ('Error calling: %s, %s' % (cmd1, b) )
      cmd1='launchctl load  %s' % matches[0]
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
        raise Exception ('The System paths have been updated. Please restart KLayout, and try again.')
  # end of OSX 

  if sys.platform.startswith('win'):
    pass
  # end of Windows
  
  
  ##################################################################
  # Load Lumerical API: 

  import lumapi_osx as lumapi
  from .. import _globals
  _globals.INTC  # Python Lumerical INTERCONNECT integration handle
  
  run_INTC()
  lumapi.evalScript(_globals.INTC, "a=0:0.01:10; plot(a,sin(a),'Congratulations, Lumerical is now available from KLayout','','Congratulations, Lumerical is now available from KLayout');")

  import os 
  # Read INTC element library
  lumapi.evalScript(_globals.INTC, "out=library;")
  INTC_libs=lumapi.getVar(_globals.INTC, "out")

  # Install technology CML if missing in INTC
  dir_path = os.path.join(pya.Application.instance().application_data_path(), 'Lumerical_CMLs')
  from ..utils import get_technology, get_technology_by_name
  # get current technology
  TECHNOLOGY = get_technology() 
  # load more technology details (CML file location)
  TECHNOLOGY = get_technology_by_name(TECHNOLOGY['technology_name'])
  # check if the latest version of the CML is in KLayout's tech
  if not ("design kits::"+TECHNOLOGY['technology_name'].lower()+"::"+TECHNOLOGY['INTC_CML_version'].replace('.cml','').lower()) in INTC_libs:
    # install CML
    print("Lumerical INTC, installdesignkit ('%s', '%s', true);" % (TECHNOLOGY['INTC_CML_path'], dir_path ) )
    lumapi.evalScript(_globals.INTC, "installdesignkit ('%s', '%s', true);" % (TECHNOLOGY['INTC_CML_path'], dir_path ) )
    # Re-Read INTC element library
    lumapi.evalScript(_globals.INTC, "out=library;")
    INTC_libs=lumapi.getVar(INTC, "out")

  # Save INTC element library to KLayout application data path
  fh = open(os.path.join(dir_path,"Lumerical_INTC_CMLs.txt"), "w")
  fh.writelines(INTC_libs)
  fh.close()


def component_simulation(verbose=False):
  import sys, os, string
  from .. import _globals
    
  # Run INTERCONNECT
  # using Lumerical API: 
  import lumapi_osx as lumapi

  # get selected instances
  from ..utils import select_instances
  selected_instances = select_instances()

  from ..utils import get_layout_variables
  TECHNOLOGY, lv, ly, cell = get_layout_variables()
    
  
  # check that it is one or more:
  error = pya.QMessageBox()
  error.setStandardButtons(pya.QMessageBox.Ok )
  if len(selected_instances) == 0:
    error.setText("Error: Need to have a component selected.")
    return
  warning = pya.QMessageBox()
  warning.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel)
  warning.setDefaultButton(pya.QMessageBox.Yes)
  if len(selected_instances) > 1 :
    warning.setText("Warning: More than one component selected.")
    warning.setInformativeText("Do you want to Proceed?")
    if(pya.QMessageBox_StandardButton(warning.exec_()) == pya.QMessageBox.Cancel):
      return
  
  # Check if the component has a compact model loaded in INTERCONNECT
  # Loop if more than one component selected
  for obj in selected_instances:
# *** not working. .returns Flattened.
#    c = obj.inst().cell.find_components()[0]
    c = cell.find_components(obj.inst().cell)[0]
    
    if not c.has_model():
      if len(selected_instances) == 0:
        error.setText("Error: Component '%s' does not have a compact model. Cannot perform simulation." % c)
        return

    # GUI to ask which pin to inject light into
    pin_names = [p.pin_name for p in c.pins if p.type == _globals.PIN_TYPES.OPTICAL or p.type == _globals.PIN_TYPES.OPTICALIO]
    pin_injection = pya.InputDialog.ask_item("Pin selection", "Choose one of the pins in component '%s' to inject light into." % c.component, pin_names, 0)
    if verbose:
      print("Pin selected from InputDialog = %s, for component '%s'." % (pin_injection, c.component) )
    
    # Write spice netlist and simulation script
    from ..utils import get_technology
    TECHNOLOGY = get_technology()  # get current technology
    import SiEPIC
    from time import strftime 
    text_main = '* Spice output from KLayout SiEPIC-Tools v%s, %s technology, %s.\n\n' % (SiEPIC.__version__, TECHNOLOGY['technology_name'], strftime("%Y-%m-%d %H:%M:%S") )
    nets_str = ''
    for p in c.pins:
      if p.type == _globals.PIN_TYPES.OPTICAL or p.type == _globals.PIN_TYPES.OPTICALIO:
        nets_str += " " + str(p.pin_name)
    # *** todo: some other way of getting this information; not hard coded.
    # GUI? Defaults from PCell?
    orthogonal_identifier=1
    wavelength_start=1500
    wavelength_stop=1600
    wavelength_points=2000
    text_main += '* Optical Network Analyzer:\n'
    text_main += '.ona input_unit=wavelength input_parameter=start_and_stop\n  + minimum_loss=80\n  + analysis_type=scattering_data\n  + multithreading=user_defined number_of_threads=1\n' 
    text_main += '  + orthogonal_identifier=%s\n' % orthogonal_identifier
    text_main += '  + start=%4.3fe-9\n' % wavelength_start
    text_main += '  + stop=%4.3fe-9\n' % wavelength_stop
    text_main += '  + number_of_points=%s\n' % wavelength_points
    for i in range(0,len(pin_names)):
      text_main += '  + input(%s)=SUBCIRCUIT,%s\n' % (i+1, pin_names[i])
    text_main += '  + output=SUBCIRCUIT,%s\n\n' % (pin_injection)
    text_main += 'SUBCIRCUIT %s SUBCIRCUIT sch_x=-1 sch_y=-1 \n\n' % (nets_str)
    text_main += '.subckt SUBCIRCUIT %s\n' % (nets_str)
    text_main += ' %s %s %s ' % ( c.component.replace(' ', '_') +"_1", nets_str, c.component.replace(' ', '_') ) 
    if c.library != None:
      text_main += 'library="%s" %s \n' % (c.library, c.params)
    text_main += '.ends SUBCIRCUIT'

    if sys.platform.startswith("win"):
      folder_name = app.application_data_path()
      if not os.path.isdir(folder_name+'/tmp'):
        os.makedirs(folder_name+"/tmp")
      filename = folder_name + '/tmp/%s_main.spi' % c.component
      filename2 = folder_name + '/tmp/%s.lsf' % c.component
      filename_icp = folder_name + '/tmp/%s.icp' % c.component
    elif sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
      import getpass
      username=getpass.getuser()
      tmp_folder='/tmp/klayout_%s_%s' % (TECHNOLOGY['technology_name'], username)
      if not os.path.isdir(tmp_folder):
        os.makedirs(tmp_folder)
      filename = '%s/%s_main.spi' % (tmp_folder, c.component)
      filename2 = '%s/%s.lsf' % (tmp_folder, c.component)
      filename_icp = '%s/%s.icp' % (tmp_folder, c.component)
    # Write the Spice netlist to file
    file = open(filename, 'w')
    file.write (text_main)
    file.close()
    if verbose:
      print(text_main)

    '''
    # Ask user whether to start a new visualizer, or use an existing one.
    opt_in_labels = [o['opt_in'] for o in opt_in]
    opt_in_labels.insert(0,'All opt-in labels')
    opt_in_selection_text = pya.InputDialog.ask_item("opt_in selection", "Choose one of the opt_in labels, to fetch experimental data.",  opt_in_labels, 0)
    if not opt_in_selection_text: # user pressed cancel
      pass
    '''    

    # Write the Lumerical INTERCONNECT start-up script.
    text_lsf =  'switchtolayout;\n'
    text_lsf += 'deleteall;\n'
    text_lsf += 'importnetlist("%s");\n' % filename
    text_lsf += 'save("%s");\n' % filename_icp
    text_lsf += 'run;\n'
    for i in range(0, len(pin_names)):
      text_lsf += 't%s = getresult("ONA_1", "input %s/mode 1/gain");\n' % (i+1, i+1)
    text_lsf += 'visualize(t1'
    for i in range(1, len(pin_names)):
      text_lsf += ', t%s' % (i+1)
    text_lsf += ');\n'
    file = open(filename2, 'w')
    file.write (text_lsf)
    file.close()
    if verbose:
      print(text_lsf)
    
    # Run using Python integration:
    from .. import _globals
    run_INTC()
    lumapi.evalScript(_globals.INTC, "cd ('" + tmp_folder + "');")
    lumapi.evalScript(_globals.INTC, c.component + ";")

    
  


def circuit_simulation():
  print('circuit_simulation()')

  import os, platform, sys, string
  print(os.name)
  print(platform.system())
  print(platform.release())
  version = sys.version
  
  # check for supported operating system, tested on:
  # Windows 7, 10
  # OSX Sierra, High Sierra
  # Linux
  if not any ( [sys.platform.startswith("win"), sys.platform.startswith("linux"), sys.platform.startswith("darwin") ]):
    raise Exception("Unsupported operating system: %s" % sys.platform)
  
  from ..utils import get_technology
  from .. import _globals
  TECHNOLOGY = get_technology()
  
  
  lv = pya.Application.instance().main_window().current_view()
  if lv == None:
    raise Exception("No view selected")
  if pya.Application.instance().main_window().current_view() == None:
    raise Exception("Please create a layout before running a simulation.")
  # find the currently selected cell:
  topcell = pya.Application.instance().main_window().current_view().active_cellview().cell
  if topcell == None:
    raise Exception("No cell")
  layout = topcell.layout()

  # Save the layout prior to running simulations, if there are changes.
  mw = pya.Application.instance().main_window()
  if mw.manager().has_undo():
    mw.cm_save()
  layout_filename = mw.current_view().active_cellview().filename()
  if len(layout_filename) == 0:
    raise Exception("Please save your layout before running the simulation")
    
  # *** todo
  '''
  # Ask user whether to trim the netlist
  # Trim Netlist setting is saved in a "User Properties" defined via the Cells window.
  # if missing, a dialog is presented.
  Trim_Netlist_options = ['Trim netlist by finding components connected to the Laser', 'Export full circuit']
  Trim_Netlist = topcell.property("Trim_Netlist")
  if Trim_Netlist and Trim_Netlist in Trim_Netlist_options:
    print("Trim_Netlist {%s}, taken from cell {%s}" % (Trim_Netlist, topcell.name) )
  else:
    Trim_Netlist = pya.InputDialog.ask_item("Trim_Netlist", "Do you want to export the complete circuit, or trim the netlist?  \nYour preference will be saved as a user property in cell {%s}." % topcell.name, Trim_Netlist_options, 0)
    if Trim_Netlist == None:
      Trim_Netlist = Trim_Netlist_options[1]
    else:
      # Record a transaction, to enable "undo"
      lv.transaction("Trim_Netlist selection")
      print("Trim_Netlist taken from the InputDialog = %s; for next time, saved in cell {%s}." % (Trim_Netlist, topcell.name) )
      topcell.set_property("Trim_Netlist", Trim_Netlist)
      lv.commit()
  '''

  # *** todo    
  #   Add the "disconnected" component to all disconnected pins
  #  optical_waveguides, optical_components = terminate_all_disconnected_pins(optical_pins, optical_waveguides, optical_components)

  '''  
  # Output the Spice netlist:
  if Trim_Netlist == Trim_Netlist_options[1]:
    text_Spice, text_Spice_main, num_detectors = generate_Spice_file(topcell, optical_waveguides, optical_components, optical_pins)
  else:
    text_Spice, text_Spice_main, num_detectors = generate_short_spice_files(topcell, optical_waveguides, optical_components, optical_pins)
  '''

  # Output the Spice netlist:
  text_Spice, text_Spice_main, num_detectors = topcell.spice_netlist_export()
  
  print(text_Spice)
  
  if sys.platform.startswith("win"):
    folder_name = app.application_data_path()
    if not os.path.isdir(folder_name+'/tmp'):
      os.makedirs(folder_name+"/tmp")
    filename = folder_name + '/tmp/%s_main.spi' % topcell.name
    filename_subckt = folder_name + '/tmp/%s.spi' % topcell.name
    filename2 = folder_name + '/tmp/%s.lsf' % topcell.name
    filename_icp = folder_name + '/tmp/%s.icp' % topcell.name
  elif sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
    import getpass
    username=getpass.getuser()
    tmp_folder='/tmp/klayout_%s_%s' % (TECHNOLOGY['technology_name'], username)
    if not os.path.isdir(tmp_folder):
      os.makedirs(tmp_folder)
    filename = '%s/%s_main.spi' % (tmp_folder, topcell.name)
    filename_subckt = '%s/%s.spi' % (tmp_folder, topcell.name)
    filename2 = '%s/%s.lsf' % (tmp_folder, topcell.name)
    filename_icp = '%s/%s.icp' % (tmp_folder, topcell.name)

  
  text_Spice_main += '.INCLUDE "%s"\n\n' % (filename_subckt)
  
  # Write the Spice netlist to file
  file = open(filename, 'w')
  file.write (text_Spice_main)
  file.close()
  file = open(filename_subckt, 'w')
  file.write (text_Spice)
  file.close()
  
  # Write the Lumerical INTERCONNECT start-up script.
  file = open(filename2, 'w')
  text_lsf = 'switchtolayout;\n'
  text_lsf += 'deleteall;\n'
  text_lsf += 'importnetlist("%s");\n' % filename
  text_lsf += 'addproperty("::Root Element::%s", "MC_uniformity_thickness", "wafer", "Matrix");\n' % topcell.name
  text_lsf += 'addproperty("::Root Element::%s", "MC_uniformity_width", "wafer", "Matrix");\n' % topcell.name
  text_lsf += 'addproperty("::Root Element::%s", "MC_grid", "wafer", "Number");\n' % topcell.name 
  text_lsf += 'addproperty("::Root Element::%s", "MC_resolution_x", "wafer", "Number");\n' % topcell.name
  text_lsf += 'addproperty("::Root Element::%s", "MC_resolution_y", "wafer", "Number");\n' % topcell.name
  text_lsf += 'addproperty("::Root Element::%s", "MC_non_uniform", "wafer", "Number");\n'  % topcell.name
  text_lsf += 'select("::Root Element::%s");\n' % topcell.name
  text_lsf += 'set("run setup script",2);\n'
  text_lsf += 'save("%s");\n' % filename_icp
  text_lsf += 'run;\n'
  for i in range(0, num_detectors):
    text_lsf += 't%s = getresult("ONA_1", "input %s/mode 1/gain");\n' % (i+1, i+1)
  text_lsf += 'visualize(t1'
  for i in range(1, num_detectors):
    text_lsf += ', t%s' % (i+1)
  text_lsf += ');\n'
  
  file.write (text_lsf)
  file.close()
  
  print(text_lsf)
  
  if sys.platform.startswith('linux'):
    # Linux-specific code here...
    if string.find(version,"2.") > -1:
      import commands
      print("Running INTERCONNECT")
      commands.getstatusoutput('/opt/lumerical/interconnect/bin/interconnect -run %s' % filename2)
  
  elif sys.platform.startswith('darwin'):
    # OSX specific

    try: 
      import lumapi_osx as lumapi
      from .. import _globals
      run_INTC()
      # Run using Python integration:
      lumapi.evalScript(_globals.INTC, "switchtolayout;")
      lumapi.evalScript(_globals.INTC, "cd ('" + tmp_folder + "');")
      lumapi.evalScript(_globals.INTC, topcell.name + ";")

    except:
      print ("Can't connect to Lumerical INTERCONNECT using Python. Proceeding using command interface.")

      if string.find(version,"2.7.") > -1:
        import commands
        home = os.path.expanduser("~")
        if ~os.path.exists(home + "/.bash_profile"):
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
        print("Running INTERCONNECT")
        runcmd = 'source ~/.bash_profile; open -n /Applications/Lumerical/INTERCONNECT/INTERCONNECT.app --args -run %s' % filename2
        print("Running in shell: %s" % runcmd)
        commands.getstatusoutput(runcmd)

    
  
  elif sys.platform.startswith('win'):
    # Windows specific code here
    import subprocess
    print("Running INTERCONNECT")
    #check Interconnect installation directory
    file_path_a = 'C:\\Program Files\\Lumerical\\INTERCONNECT\\bin\\interconnect.exe'
    file_path_b = 'C:\\Program Files (x86)\\Lumerical\\INTERCONNECT\\bin\\interconnect.exe'
    
    if(os.path.isfile(file_path_a)==True):
      subprocess.Popen(args=[file_path_a, '-run', filename2], shell=True)
      
    elif(os.path.isfile(file_path_b)==True):
      subprocess.Popen(args=[file_path_b, '-run', filename2], shell=True)
    
    else:
      warning_window = pya.QMessageBox()
      warning_window.setText("Warning: The program could not find INTERCONNECT.")
      warning_window.setInformativeText("Do you want to specify it manually?")
      warning_window.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel);
      warning_window.setDefaultButton(pya.QMessageBox.Yes)
      response = warning_window.exec_()        
      if(response == pya.QMessageBox.Yes):
        dialog = pya.QFileDialog()
        path = str(dialog.getOpenFileName())
        path = path.replace('/', '\\')
        subprocess.Popen(args=[path, '-run', filename2], shell=True)



  
def circuit_simulation_update_netlist():
  print('update netlist')
  
  
def circuit_simulation_monte_carlo(params = None, cell = None):
  from .. import _globals

  lv = pya.Application.instance().main_window().current_view()
  if lv == None:
    raise Exception("No view selected")

  if cell is None:
    ly = lv.active_cellview().layout() 
    if ly == None:
      raise Exception("No active layout")
    cell = lv.active_cellview().cell
    if cell == None:
      raise Exception("No active cell")
  else:
    ly = cell.layout()
  
  status = _globals.MC_GUI.return_status()
  if status is None and params is None:
    _globals.MC_GUI.show()
  else:
    if status is False: return
    if params is None: params = _globals.MC_GUI.get_parameters()
    
  print("monte_carlo")