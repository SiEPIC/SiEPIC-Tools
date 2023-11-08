'''
################################################################################
#
#  SiEPIC-Tools
#  
################################################################################

Circuit simulations using Lumerical INTERCONNECT and a Compact Model Library

- run_INTC: run INTERCONNECT using Python integration
- INTC_commandline: invoke INTC via the command line, with an lsf file as input.
- Setup_Lumerical_KLayoutPython_integration
    Configure PATH env, import lumapi, run interconnect, 
    Install technology CML, read CML elements
- circuit_simulation: netlist extract and run simulation
- circuit_simulation_update_netlist: update netlist and run simulation
- circuit_simulation_monte_carlo: perform many simulations
- component_simulation: single component simulation

usage:
 import SiEPIC.lumerical.interconnect


################################################################################
'''



import pya

def run_INTC(verbose=False):
  from . import load_lumapi
  from .. import _globals
  lumapi = _globals.LUMAPI
  if not lumapi:
    print("SiEPIC.lumerical.interconnect.run_INTC: lumapi not loaded; reloading load_lumapi.")
    import sys
    if sys.version_info[0] == 3:
        if sys.version_info[1] < 4:
            from imp import reload
        else:
            from importlib import reload
    elif sys.version_info[0] == 2:
        from imp import reload    
    reload(load_lumapi)

  if not lumapi:
    print("SiEPIC.lumerical.interconnect.run_INTC: lumapi not loaded")
    pya.MessageBox.warning("Cannot load Lumerical Python integration.", "Cannot load Lumerical Python integration. \nSome SiEPIC-Tools Lumerical functionality will not be available.", pya.MessageBox.Cancel)
    return
  
  if verbose:
    print(_globals.INTC)  # Python Lumerical INTERCONNECT integration handle
  
  if not _globals.INTC: # Not running, start a new session
    _globals.INTC = lumapi.open('interconnect')
    if verbose:
      print(_globals.INTC)  # Python Lumerical INTERCONNECT integration handle
  else: # found open INTC session
    try:
      lumapi.evalScript(_globals.INTC, "?'KLayout integration test.\n';\n")
    except: # but can't communicate with INTC; perhaps it was closed by the user
      _globals.INTC = lumapi.open('interconnect') # run again.
      if verbose:
        print(_globals.INTC)  # Python Lumerical INTERCONNECT integration handle
  try: # check again
    lumapi.evalScript(_globals.INTC, "?'KLayout integration test.\n';\n")
  except:
    raise Exception ("Can't run Lumerical INTERCONNECT via Python integration.")


   

def INTC_installdesignkit(verbose=False):
  '''Install a Compact Model Library into Lumerical INTERCONNECT,
  using a compressed (.cml) in the PDK, placing it in the KLayout folder.'''

  import sys, os, string, pya
  from ..utils import get_technology, get_technology_by_name

  # get current technology
  TECHNOLOGY = get_technology(query_activecellview_technology=True)  
  # load more technology details (CML file location)
  TECHNOLOGY = get_technology_by_name(TECHNOLOGY['technology_name'])
  
  # location for the where the CMLs will locally be installed:
  dir_path = os.path.join(pya.Application.instance().application_data_path(), 'Lumerical_CMLs')

  try:
    libraries = []
    library_ids = [n for n in pya.Library.library_ids() if (pya.Library.library_by_id(n).technology == TECHNOLOGY['technology_name'])]
    for n in [pya.Library.library_by_id(l) for l in library_ids]:
      print(n.layout().meta_info_value("path"))
      libraries.append ( n.name() )
  except:
    pass

  if 'INTC_CMLs_path' in TECHNOLOGY:
    question = pya.QMessageBox()
    question.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.No)
    question.setDefaultButton(pya.QMessageBox.Yes)
    question.setText("SiEPIC-Tools will install the Compact Model Library (CML) in Lumerical INTERCONNECT for the currently active technology. \nThis includes the libraries %s.  \nProceed?" % libraries)
    informative_text = "\nTechnology: %s\n" % TECHNOLOGY['technology_name']
    for i in range(0,len(TECHNOLOGY['INTC_CMLs_path'])):
      informative_text += "Source CML file {}: {}\n".format(i+1, TECHNOLOGY['INTC_CMLs_path'][i])
    informative_text += "Install location: %s" % dir_path
    question.setInformativeText(informative_text)
    if(pya.QMessageBox_StandardButton(question.exec_()) == pya.QMessageBox.No):
      return

  
  ##################################################################
  # Load Lumerical API: 
  from .. import _globals
  run_INTC()
  lumapi = _globals.LUMAPI
  if not lumapi:
    print('SiEPIC.lumerical.interconnect.Setup_Lumerical_KLayoutPython_integration: lumapi not loaded')
    return

  import os 
  # Read INTC element library
  lumapi.evalScript(_globals.INTC, "out=library;")
  _globals.INTC_ELEMENTS=lumapi.getVar(_globals.INTC, "out")

  # Install other CMLs for each library that has a path folder
  for i in range(0,len(library_ids)):
    print(' - checking for library: %s' % (pya.Library.library_by_id(library_ids[i]).name()) )
    if 'path' in dir(pya.Library.library_by_id(library_ids[i])):
        path = pya.Library.library_by_id(library_ids[i]).path
        # look for CML files
        print(' - checking for CMLs: %s' % (path) )
        for f in os.listdir(path):
            if f.lower().endswith(".cml"):
                ff = os.path.join(path,f)
                print(' - CML file: %s' % (ff) )
                # install CML
                print("Lumerical INTC, installdesignkit ('%s', '%s', true);" % (ff, dir_path ) )
                lumapi.evalScript(_globals.INTC, "installdesignkit ('%s', '%s', true);" % (ff, dir_path ) )

  # Install technology CML if missing in INTC
  # check if the latest version of the CML is in KLayout's tech
  if 'INTC_CMLs_path' in TECHNOLOGY and TECHNOLOGY['INTC_CML_path']:
    if not ("design kits::"+TECHNOLOGY['technology_name'].lower()+"::"+TECHNOLOGY['INTC_CML_version'].lower().replace('.cml','').lower()) in _globals.INTC_ELEMENTS:
      # install CML
      print("Lumerical INTC, installdesignkit ('%s', '%s', true);" % (TECHNOLOGY['INTC_CML_path'], dir_path ) )
      lumapi.evalScript(_globals.INTC, "installdesignkit ('%s', '%s', true);" % (TECHNOLOGY['INTC_CML_path'], dir_path ) )
    
    # Install other CMLs within technology
    for i in range(0,len(TECHNOLOGY['INTC_CMLs_name'])):
      if not ("design kits::"+TECHNOLOGY['INTC_CMLs_name'][i].lower()+"::"+TECHNOLOGY['INTC_CMLs_version'][i].lower().replace('.cml','').lower()) in _globals.INTC_ELEMENTS:
        # install CML
        print("Lumerical INTC, installdesignkit ('%s', '%s', true);" % (TECHNOLOGY['INTC_CMLs_path'][i], dir_path ) )
        lumapi.evalScript(_globals.INTC, "installdesignkit ('%s', '%s', true);" % (TECHNOLOGY['INTC_CMLs_path'][i], dir_path ) )
        
  # Re-Read INTC element library
  lumapi.evalScript(_globals.INTC, "out=library;")
  _globals.INTC_ELEMENTS=lumapi.getVar(_globals.INTC, "out")
  # Close INTERCONNECT so that the library information is saved, then re-open
  lumapi.close(_globals.INTC)
  run_INTC()

  # Save INTC element library to KLayout application data path
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  fh = open(os.path.join(dir_path,"Lumerical_INTC_CMLs.txt"), "w")
  fh.writelines(_globals.INTC_ELEMENTS)
  fh.close()
  
  integration_success_message = "message('KLayout-Lumerical INTERCONNECT integration successful, CML library/libraries:\n"
  for cml_name in TECHNOLOGY['INTC_CMLs_name']:
    integration_success_message += "design kits::"+cml_name.lower()+"\n"
  integration_success_message += "');switchtodesign;\n"
  lumapi.evalScript(_globals.INTC, integration_success_message)

  # instantiate all library elements onto the canvas
  question = pya.QMessageBox()
  question.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.No)
  question.setDefaultButton(pya.QMessageBox.Yes)
  question.setText("Do you wish to see all the components in %s library?" % TECHNOLOGY['technology_name'])
#  question.setInformativeText("Do you wish to see all the components in the library?")
  if(pya.QMessageBox_StandardButton(question.exec_()) == pya.QMessageBox.No):
    # lumapi.evalScript(_globals.INTC, "b=0:0.01:10; plot(b,sin(b),'Congratulations, Lumerical is now available from KLayout','','Congratulations, Lumerical is now available from KLayout');")
    return
  intc_elements = _globals.INTC_ELEMENTS.split('\n')
#  tech_elements = [ e.split('::')[-1] for e in intc_elements if "design kits::"+TECHNOLOGY['technology_name'].lower()+"::" in e ]
  tech_elements = [ e for e in intc_elements if "design kits::"+TECHNOLOGY['technology_name'].lower()+"::" in e ]
  i, x, y, num = 0, 0, 0, len(tech_elements)
  for i in range(0, num):
    lumapi.evalScript(_globals.INTC, "a=addelement('%s'); setposition(a,%s,%s); " % (tech_elements[i],x,y) )
    y += 250
    if (i+1) % int(num**0.5) == 0:
      x += 250
      y = 0

def INTC_loaddesignkit(folder_CML, verbose=False):
    '''Load a Compact Model Library into Lumerical INTERCONNECT,
    using a folder in the PDK.'''

    import os

    # check if there are CML folders:
    if not next(os.walk(folder_CML))[1]:
        raise Exception ('SiEPIC.lumerical.interconnect.INTC_loaddesignkit: Cannot find Lumerical INTERCONNECT Compact Model Library (CML) folder, which should be in the PDK folder under the CML subfolder.')
        

    # Load Lumerical INTERCONNECT and Python API: 
    from .. import _globals
    run_INTC()
    lumapi = _globals.LUMAPI
    if not lumapi:
        raise Exception ('SiEPIC.lumerical.interconnect.INTC_loaddesignkit: Cannot load Lumerical INTERCONNECT and Python integration (lumapi).')

    # Read INTC element library
    lumapi.evalScript(_globals.INTC, "out=library;")
    _globals.INTC_ELEMENTS=lumapi.getVar(_globals.INTC, "out")

    # Load all the subfolders of the CML folder into INTERCONNECT:
    folder_names = []
    new_loaded = False
    for folder_name in next(os.walk(folder_CML))[1]:
        if not ("design kits::"+folder_name.lower()+"::" in _globals.INTC_ELEMENTS):
            folder_path = os.path.join(folder_CML,folder_name)
            lumapi.evalScript(_globals.INTC, "loaddesignkit ('%s', '%s');" % (folder_name, folder_path ) )
            new_loaded = True
        folder_names.append(folder_name)

    # Close INTERCONNECT so that the library information is saved, then re-open
    if new_loaded:
        lumapi.close(_globals.INTC)
        run_INTC()

    # Read INTC element library
    lumapi.evalScript(_globals.INTC, "out=library;")
    _globals.INTC_ELEMENTS=lumapi.getVar(_globals.INTC, "out")
    print('%s' % folder_names)

    if  "design kits::"+folder_name.lower()+"::" in _globals.INTC_ELEMENTS:
        integration_success_message = "message('KLayout-Lumerical INTERCONNECT integration successful, for Compact Model Libraries installed in Element Library | Design kits: %s.\n" % (' '.join(map(str,folder_names)))
        integration_success_message += "');switchtodesign;\n"
        lumapi.evalScript(_globals.INTC, integration_success_message)


def Setup_Lumerical_KLayoutPython_integration(verbose=False):
    '''Setup Lumerical INTERCONNECT KLayout integration,
    including loading/installing a Compact Model Library.
    Requires:
        - an open window with an active technology
        - a compact model library, sourced from the Technology folder 
            (where the *.lyt file is located)
            either, and in priority:
            - inside the folder "CML"
            - *.CML files
    '''

    # Check if there is a layout open, so we know which technology to install
    lv = pya.Application.instance().main_window().current_view()
    if lv == None:
        raise UserWarning("To install the Compact Model Library, first, please create a new layout and select the desired technology:\n  Menu: File > New Layout, and a Technology.\nThen repeat.")

  
    # Get the Technology 
    from SiEPIC.utils import get_layout_variables
    TECHNOLOGY, lv, ly, top_cell = get_layout_variables()
    
    # Check if there is a CML folder in the Technology folder
    import os
    base_path = ly.technology().base_path()    
    folder_CML = os.path.join(base_path,'CML')
    if os.path.exists(folder_CML):
        print('Setup_Lumerical_KLayoutPython_integration: Load design kit')
        return INTC_loaddesignkit(folder_CML)
    else:
        print('Setup_Lumerical_KLayoutPython_integration: Install design kit')
        return INTC_installdesignkit()
    

def INTC_commandline(filename2):
  print ("Running Lumerical INTERCONNECT using the command interface.")
  import sys, os, string
  
  if sys.platform.startswith('linux'):
    import subprocess
    # Linux-specific code here...
    print("Running INTERCONNECT")
    # Location of INTERCONNECT program (this found from RPM installation)
    file_path = '/opt/lumerical/interconnect/bin/interconnect'
    subprocess.Popen([file_path, '-run', filename2])
      
  
  elif sys.platform.startswith('darwin'):
    # OSX specific
    import sys
    if int(sys.version[0]) > 2:
      import subprocess
      subprocess.Popen(['/usr/bin/open -n /Applications/Lumerical/INTERCONNECT/INTERCONNECT.app', '-run', '--args -run %s' % filename2])          
    else:
      import commands
      print("Running INTERCONNECT")
      runcmd = ('source ~/.bash_profile; /usr/bin/open -n /Applications/Lumerical/INTERCONNECT/INTERCONNECT.app --args -run %s' % filename2)
      print("Running in shell: %s" % runcmd)
      a=commands.getstatusoutput(runcmd)
      print(a)

  
  elif sys.platform.startswith('win'):
    # Windows specific code here
    import subprocess
    print("Running INTERCONNECT")
    #check Interconnect installation directory
    file_path_a = 'C:\\Program Files\\Lumerical\\INTERCONNECT\\bin\\interconnect.exe'
    file_path_b = 'C:\\Program Files (x86)\\Lumerical\\INTERCONNECT\\bin\\interconnect.exe'
    file_path_c = 'C:\\Program Files\\Lumerical\\v202\\bin\\interconnect.exe'
    if(os.path.isfile(file_path_a)==True):
      subprocess.Popen(args=[file_path_a, '-run', filename2], shell=True)
    elif(os.path.isfile(file_path_b)==True):
      subprocess.Popen(args=[file_path_b, '-run', filename2], shell=True)
    elif(os.path.isfile(file_path_c)==True):
      subprocess.Popen(args=[file_path_c, '-run', filename2], shell=True)
    else:
      warning_window = pya.QMessageBox()
      warning_window.setText("Warning: The program could not find INTERCONNECT.")
      warning_window.setInformativeText("Do you want to specify it manually?")
      warning_window.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel);
      warning_window.setDefaultButton(pya.QMessageBox.Yes)
      response = warning_window.exec_()        
      if(pya.QMessageBox_StandardButton(response) == pya.QMessageBox.Yes):
        dialog = pya.QFileDialog()
        path = str(dialog.getOpenFileName())
        path = path.replace('/', '\\')
        subprocess.Popen(args=[path, '-run', filename2], shell=True)


def component_simulation(verbose=False, simulate=True):
  import sys, os, string
  from .. import _globals

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
    if verbose:
      print("  selected component: %s" % obj.inst().cell )
    c = cell.find_components(cell_selected=[obj.inst().cell])
    if c:
      c=c[0]
    else:
      continue
    
    if not c.has_model():
      if len(selected_instances) == 0:
        error.setText("Error: Component '%s' does not have a compact model. Cannot perform simulation." % c)
        continue

    # GUI to ask which pin to inject light into
    pin_names = [p.pin_name for p in c.pins if p.type == _globals.PIN_TYPES.OPTICAL or p.type == _globals.PIN_TYPES.OPTICALIO]
    if not pin_names:
      continue
    pin_injection = pya.InputDialog.ask_item("Pin selection", "Choose one of the pins in component '%s' to inject light into." % c.component, pin_names, 0)
    if not pin_injection:
      return
    if verbose:
      print("Pin selected from InputDialog = %s, for component '%s'." % (pin_injection, c.component) )
    
    # Write spice netlist and simulation script
    from ..utils import get_technology
    TECHNOLOGY = get_technology()  # get current technology
    import SiEPIC
    from time import strftime 
    text_main = '* Spice output from KLayout SiEPIC-Tools v%s, %s technology (SiEPIC.lumerical.interconnect.component_simulation), %s.\n\n' % (SiEPIC.__version__, TECHNOLOGY['technology_name'], strftime("%Y-%m-%d %H:%M:%S") )



    # find electrical IO pins
    electricalIO_pins = ""
    DCsources = "" # string to create DC sources for each pin
    Vn = 1
    # (2) or to individual DC sources
    # create individual sources:
    for p in c.pins:
      if p.type == _globals.PIN_TYPES.ELECTRICAL:
        NetName = " " + c.component +'_' + str(c.idx) + '_' + p.pin_name
        electricalIO_pins += NetName
        DCsources += 'N' + str(Vn) + NetName + ' "dc source" amplitude=0 sch_x=%s sch_y=%s\n' % (-2-Vn/3., -2+Vn/8.)
        Vn += 1
    electricalIO_pins_subckt = electricalIO_pins


    # component nets: must be ordered electrical, optical IO, then optical
    nets_str = ''
    DCsources = "" # string to create DC sources for each pin
    Vn = 1
    for p in c.pins: 
      if p.type == _globals.PIN_TYPES.ELECTRICAL:
        if not p.pin_name:
          continue
        NetName = " " + c.component +'_' + str(c.idx) + '_' + p.pin_name
        nets_str += NetName
        DCsources += "N" + str(Vn) + NetName + " dcsource amplitude=0 sch_x=%s sch_y=%s\n" % (-2-Vn/3., -2+Vn/8.)
        Vn += 1
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

    text_main += DCsources

    text_main += 'SUBCIRCUIT %s SUBCIRCUIT sch_x=-1 sch_y=-1 \n\n' % (nets_str)
    text_main += '.subckt SUBCIRCUIT %s\n' % (nets_str)
    text_main += ' %s %s %s ' % ( c.component.replace(' ', '_') +"_1", nets_str, c.component.replace(' ', '_') ) 
    if c.library != None:
      text_main += 'library="%s" %s ' % (c.library, c.params)
    text_main += '\n.ends SUBCIRCUIT\n'

    from .. import _globals
    tmp_folder = _globals.TEMP_FOLDER
    import os    
    filename = os.path.join(tmp_folder, '%s_main.spi' % c.component)
    filename2 = os.path.join(tmp_folder, '%s.lsf' % c.component)
    filename_icp = os.path.join(tmp_folder, '%s.icp' % c.component)

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
    text_lsf +=  "cd('%s');\n" % tmp_folder
    text_lsf += 'deleteall;\n'
    text_lsf += "importnetlist('%s');\n" % filename
    text_lsf += "save('%s');\n" % filename_icp
    text_lsf += 'run;\n'
    if 0:
      for i in range(0, len(pin_names)):
        text_lsf += 'h%s = haveresult("ONA_1", "input %s/mode 1/gain");\n' % (i+1, i+1)
        text_lsf += 'if (h%s>0) { visualize(getresult("ONA_1", "input %s/mode 1/gain")); } \n' % (i+1, i+1)
    if 1:
      text_lsf += 't = "";\n'
      for i in range(0, len(pin_names)):
        text_lsf += 'h%s = haveresult("ONA_1", "input %s/mode 1/gain");\n' % (i+1, i+1)
        text_lsf += 'if (h%s>0) { t%s = getresult("ONA_1", "input %s/mode 1/gain"); t=t+"t%s,"; } \n' % (i+1, i+1, i+1, i+1)
      text_lsf += 't = substring(t, 1, length(t) - 1);\n'
      text_lsf += 'eval("visualize(" + t + ");");\n'

    file = open(filename2, 'w')
    file.write (text_lsf)
    file.close()
    if verbose:
      print(text_lsf)
    
    if simulate:
      # Run using Python integration:
      try: 
        from .. import _globals
        run_INTC()
        # Run using Python integration:
        lumapi = _globals.LUMAPI
        lumapi.evalScript(_globals.INTC, "cd ('" + tmp_folder + "');")
        lumapi.evalScript(_globals.INTC, "feval('"+ c.component + "');\n")
      except:
        from .. import scripts
        scripts.open_folder(tmp_folder)
        INTC_commandline(filename)
    else:
      from .. import scripts
      scripts.open_folder(tmp_folder)

def circuit_simulation_toolbar():
  circuit_simulation(verbose=False,opt_in_selection_text=[], matlab_data_files=[], simulate=True)

def circuit_simulation(verbose=False,opt_in_selection_text=[], matlab_data_files=[], simulate=True):
  print ('*** circuit_simulation(), opt_in: %s' % opt_in_selection_text)
  if verbose:
    print('*** circuit_simulation()')
  
  # check for supported operating system, tested on:
  # Windows 7, 10
  # OSX Sierra, High Sierra
  # Linux
  import sys
  if not any([sys.platform.startswith(p) for p in {"win","linux","darwin"}]):
    raise Exception("Unsupported operating system: %s" % sys.platform)
  
  from .. import _globals
  from SiEPIC.utils import get_layout_variables
  TECHNOLOGY, lv, layout, topcell = get_layout_variables()
  
  # Save the layout prior to running simulations, if there are changes.
  mw = pya.Application.instance().main_window()
  if mw.manager().has_undo():
    mw.cm_save()
  layout_filename = mw.current_view().active_cellview().filename()
  if len(layout_filename) == 0:
    raise Exception("Please save your layout before running the simulation")
    
  # *** todo    
  #   Add the "disconnected" component to all disconnected pins
  #  optical_waveguides, optical_components = terminate_all_disconnected_pins()

  # Output the Spice netlist:
  text_Spice, text_Spice_main, num_detectors, detector_list = \
    topcell.spice_netlist_export(verbose=verbose, opt_in_selection_text=opt_in_selection_text)
  if not text_Spice:
    raise Exception("No netlist available. Cannot run simulation.")
    return
  if verbose:   
    print(text_Spice)

  circuit_name = topcell.name.replace('.','') # remove "."
  if '_' in circuit_name[0]:
    circuit_name = ''.join(circuit_name.split('_', 1))  # remove leading _
  
  from .. import _globals
  tmp_folder = _globals.TEMP_FOLDER
  import os
  filename = os.path.join(tmp_folder, '%s_main.spi' % circuit_name)
  filename_subckt = os.path.join(tmp_folder,  '%s.spi' % circuit_name)
  filename2 = os.path.join(tmp_folder, '%s.lsf' % circuit_name)
  filename_icp = os.path.join(tmp_folder, '%s.icp' % circuit_name)
  
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
  text_lsf += "importnetlist('%s');\n" % filename
  text_lsf += 'addproperty("::Root Element", "MC_uniformity_thickness", "wafer", "Matrix");\n'
  text_lsf += 'addproperty("::Root Element", "MC_uniformity_width", "wafer", "Matrix");\n' 
  text_lsf += 'addproperty("::Root Element", "MC_grid", "wafer", "Number");\n' 
  text_lsf += 'addproperty("::Root Element", "MC_resolution_x", "wafer", "Number");\n' 
  text_lsf += 'addproperty("::Root Element", "MC_resolution_y", "wafer", "Number");\n' 
  text_lsf += 'addproperty("::Root Element", "MC_non_uniform", "wafer", "Number");\n'  
  text_lsf += 'select("::Root Element");\n' 
  text_lsf += 'set("run setup script",2);\n'
  text_lsf += "save('%s');\n" % filename_icp
  text_lsf += 'run;\n'
  for i in range(1, num_detectors+1):
    if matlab_data_files:
      # convert simulation data into simple datasets:
      wavelenth_scale = 1e9
      text_lsf += 'temp = getresult("ONA_1", "input %s/mode 1/gain");\n' % i
      text_lsf += 't%s = matrixdataset("Simulation");\n' % i
      text_lsf += 't%s.addparameter("wavelength",temp.wavelength*%s);\n' % (i, wavelenth_scale)
      text_lsf += 't%s.addattribute("Simulation, Detector %s",getresultdata("ONA_1", "input %s/mode 1/gain"));\n' % (i,i, i)
    else:
      text_lsf += 't%s = getresult("ONA_1", "input %s/mode 1/gain");\n' % (i, i)
      
  # load measurement data files
  m_count=0
  if matlab_data_files:
    for m in matlab_data_files:
      if '.mat' in m:
        m_count += 1

        # *** todo, use DFT rules to determine which measurements we should load.
        
        # INTERCONNECT load data
        head, tail = os.path.split(m)
        tail = tail.split('.mat')[0]
        text_lsf += 'matlabload("%s", scandata);\n' % m
        text_lsf += 'm%s = matrixdataset("Measurement");\n' % m_count
        text_lsf += 'm%s.addparameter("wavelength",scandata.wavelength*%s);\n'  % (m_count, wavelenth_scale)
        for d in detector_list:
          text_lsf += 'm%s.addattribute("Measured: %s",scandata.power(:,%s));\n'  % (m_count, tail, d)
  
  text_lsf += 'visualize(t1'
  for i in range(2, num_detectors+1):
    text_lsf += ', t%s' % i
  for i in range(1, m_count+1):
    text_lsf += ', m%s' % i
  text_lsf += ');\n'
  
  file.write (text_lsf)
  file.close()
  
  if verbose:
    print(text_lsf)

  if simulate:
    # Run using Python integration:
    try: 
      from .. import _globals
      run_INTC()
      # Run using Python integration:
      lumapi = _globals.LUMAPI
      lumapi.evalScript(_globals.INTC, "?'';")
    except:
      from .. import scripts
      scripts.open_folder(tmp_folder)
      INTC_commandline(filename)
      print('SiEPIC.lumerical.interconnect: circuit_simulation: error 1')
    try:
      lumapi.evalScript(_globals.INTC, "cd ('" + tmp_folder + "');\n")
      print("feval('"+ circuit_name + "');\n")
      lumapi.evalScript(_globals.INTC, "feval('"+ circuit_name + "');\n")
    except:
      print('SiEPIC.lumerical.interconnect: circuit_simulation: error 2')
      pass
  else:
    from .. import scripts
    scripts.open_folder(tmp_folder)
    
  if verbose:
    print('Done Lumerical INTERCONNECT circuit simulation.')

  
def circuit_simulation_update_netlist():
  print('update netlist')
  
    
def circuit_simulation_monte_carlo(params = None, topcell = None, verbose=True, opt_in_selection_text=[], matlab_data_files=[], simulate=True):
  print('*** circuit_simulation_monte_carlo()')
  from .. import _globals
  from ..utils import get_layout_variables
  if topcell is None:
    TECHNOLOGY, lv, ly, topcell = get_layout_variables()
  else:
    TECHNOLOGY, lv, _, _ = get_layout_variables()
    ly = topcell.layout()
  
  if params is None: params = _globals.MC_GUI.get_parameters()
  if params is None: 
    pya.MessageBox.warning("No MC parameters", "No Monte Carlo parameters. Cancelling.", pya.MessageBox.Cancel)
    return
  print(params)
  
  if int(params['num_wafers'])<1:
    pya.MessageBox.warning("Insufficient number of wafers", "The number of wafers for Monte Carlo simulations need to be 1 or more.", pya.MessageBox.Cancel)
    return
  if int(params['num_dies'])<1:
    pya.MessageBox.warning("Insufficient number of dies", "The number of die per wafer for Monte Carlo simulations need to be 1 or more.", pya.MessageBox.Cancel)
    return

  circuit_name = topcell.name.replace('.','') # remove "."
  if '_' in circuit_name[0]:
    circuit_name = ''.join(circuit_name.split('_', 1))  # remove leading _
  
  
  if verbose:
    print('*** circuit_simulation_monte_carlo()')
  
  # check for supported operating system, tested on:
  # Windows 7, 10
  # OSX Sierra, High Sierra
  # Linux
  import sys
  if not any([sys.platform.startswith(p) for p in {"win","linux","darwin"}]):
    raise Exception("Unsupported operating system: %s" % sys.platform)
    
  # Save the layout prior to running simulations, if there are changes.
  mw = pya.Application.instance().main_window()
  if mw.manager().has_undo():
    mw.cm_save()
  layout_filename = mw.current_view().active_cellview().filename()
  if len(layout_filename) == 0:
    pya.MessageBox.warning("Please save your layout before running the simulation.", "Please save your layout before running the simulation.", pya.MessageBox.Cancel)
    return
    
  # *** todo    
  #   Add the "disconnected" component to all disconnected pins
  #  optical_waveguides, optical_components = terminate_all_disconnected_pins()

  # Output the Spice netlist:
  text_Spice, text_Spice_main, num_detectors, detector_list = \
    topcell.spice_netlist_export(verbose=verbose, opt_in_selection_text=opt_in_selection_text)
  if not text_Spice:
    pya.MessageBox.warning("No netlist available.", "No netlist available. Cannot run simulation.", pya.MessageBox.Cancel)
    return
  if verbose:   
    print(text_Spice)
  
  tmp_folder = _globals.TEMP_FOLDER
  import os    
  filename = os.path.join(tmp_folder, '%s_main.spi' % circuit_name)
  filename_subckt = os.path.join(tmp_folder,  '%s.spi' % circuit_name)
  filename2 = os.path.join(tmp_folder, '%s.lsf' % circuit_name)
  filename_icp = os.path.join(tmp_folder, '%s.icp' % circuit_name)
  
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

  text_lsf = '###DEVELOPER:Zeqin Lu, zqlu@ece.ubc.ca, University of British Columbia \n' 
  text_lsf += 'switchtolayout;\n'
  text_lsf += 'deleteall;\n'
  text_lsf += "importnetlist('%s');\n" % filename
  text_lsf += 'addproperty("::Root Element", "wafer_uniformity_thickness", "wafer", "Matrix");\n' 
  text_lsf += 'addproperty("::Root Element", "wafer_uniformity_width", "wafer", "Matrix");\n' 
  text_lsf += 'addproperty("::Root Element", "N", "wafer", "Number");\n'  
  text_lsf += 'addproperty("::Root Element", "selected_die", "wafer", "Number");\n' 
  text_lsf += 'addproperty("::Root Element", "wafer_length", "wafer", "Number");\n'   
  text_lsf += 'addproperty("::Root Element::%s", "MC_uniformity_thickness", "wafer", "Matrix");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_uniformity_width", "wafer", "Matrix");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_grid", "wafer", "Number");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_resolution_x", "wafer", "Number");\n'  % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_resolution_y", "wafer", "Number");\n' % circuit_name
  text_lsf += 'addproperty("::Root Element::%s", "MC_non_uniform", "wafer", "Number");\n'  % circuit_name
  text_lsf += 'select("::Root Element::%s");\n'  % circuit_name
  text_lsf += 'set("MC_non_uniform",99);\n'  
  text_lsf += 'n_wafer = %s;  \n'  % params['num_wafers']  #  GUI INPUT: Number of testing wafer
  text_lsf += 'n_die = %s;  \n'  % params['num_dies']  #  GUI INPUT: Number of testing die per wafer
  text_lsf += 'kk = 1;  \n'
  text_lsf += 'select("ONA_1");\n'
  text_lsf += 'num_points = get("number of points");\n'
  
  for i in range(0, num_detectors):
    text_lsf += 'mc%s = matrixdataset("mc%s"); # initialize visualizer data, mc%s \n' % (i+1, i+1, i+1)
    text_lsf += 'Gain_Data_input%s = matrix(num_points,n_wafer*n_die);  \n' % (i+1) 

  ###Define histograms datasets
  if(params['histograms']['fsr']==True):
    text_lsf += 'fsr_dataset = matrix(1,n_wafer*n_die,1);\n'
  if(params['histograms']['wavelength']==True):
    text_lsf += 'freq_dataset = matrix(1,n_wafer*n_die,1);\n'
  if(params['histograms']['gain']==True):
    text_lsf += 'gain_dataset = matrix(1,n_wafer*n_die,1);\n'
  
  text_lsf += '#Run Monte Carlo simulations; \n'
  text_lsf += 'for (jj=1; jj<=n_wafer; jj=jj+1) {   \n'
  ############################## Wafer generation ###########################################
  text_lsf += ' wafer_length = %s;  \n'  % 100e-3 # datadict["wafer_length_x"]  # [m], GUI INPUT: wafer length
  text_lsf += ' wafer_cl_width = %s;  \n' % params['waf_var']['width']['corr_len']  # [m],  GUI INPUT: wafer correlation length
  text_lsf += ' wafer_cl_thickness = %s;  \n' % params['waf_var']['height']['corr_len']  # [m],  GUI INPUT: wafer correlation length  
  text_lsf += ' wafer_clx_width = wafer_cl_width;  \n'  
  text_lsf += ' wafer_cly_width = wafer_cl_width; \n'   
  text_lsf += ' wafer_clx_thickness = wafer_cl_thickness;  \n'  
  text_lsf += ' wafer_cly_thickness = wafer_cl_thickness; \n'  
  text_lsf += ' N = 500;  \n'        
  text_lsf += ' wafer_grid=wafer_length/N; \n'   
  text_lsf += ' wafer_RMS_w = %s;     \n' % params['waf_var']['width']['std_dev'] # [nm], GUI INPUT: Within wafer Sigma RMS for width
  text_lsf += ' wafer_RMS_t = %s;   \n' % params['waf_var']['height']['std_dev']    # [nm], GUI INPUT: Within wafer Sigma RMS for thickness
  text_lsf += ' x = linspace(-wafer_length/2,wafer_length/2,N); \n'
  text_lsf += ' y = linspace(-wafer_length/2,wafer_length/2,N); \n'
  text_lsf += ' xx = meshgridx(x,y) ;  \n'
  text_lsf += ' yy = meshgridy(x,y) ;  \n'
  text_lsf += ' wafer_Z_thickness = wafer_RMS_t*randnmatrix(N,N);  \n'
  text_lsf += ' wafer_F_thickness = exp(-(xx^2/(wafer_clx_thickness^2/2)+yy^2/(wafer_cly_thickness^2/2))); \n'  # Gaussian filter
  text_lsf += ' wafer_uniformity_thickness = real( 2/sqrt(pi)*wafer_length/N/sqrt(wafer_clx_thickness)/sqrt(wafer_cly_thickness)*invfft(fft(wafer_Z_thickness,1,0)*fft(wafer_F_thickness,1,0), 1, 0)  );    \n' # wafer created using Gaussian filter   
  text_lsf += ' wafer_Z_width = wafer_RMS_w*randnmatrix(N,N);  \n'
  text_lsf += ' wafer_F_width = exp(-(xx^2/(wafer_clx_width^2/2)+yy^2/(wafer_cly_width^2/2))); \n'  # Gaussian filter
  text_lsf += ' wafer_uniformity_width = real( 2/sqrt(pi)*wafer_length/N/sqrt(wafer_clx_width)/sqrt(wafer_cly_width)*invfft(fft(wafer_Z_width,1,0)*fft(wafer_F_width,1,0), 1, 0)  );    \n' # wafer created using Gaussian filter 
  
  ######################## adjust Wafer mean ###################
  text_lsf += ' mean_RMS_w = %s;     \n' % params['waf_to_waf_var']['width']['std_dev'] # [nm], GUI INPUT:  wafer Sigma RMS for width
  text_lsf += ' mean_RMS_t = %s;   \n' % params['waf_to_waf_var']['thickness']['std_dev']    # [nm], GUI INPUT:  wafer Sigma RMS for thickness
  text_lsf += ' wafer_uniformity_thickness = wafer_uniformity_thickness + randn(0,mean_RMS_t); \n'
  text_lsf += ' wafer_uniformity_width = wafer_uniformity_width + randn(0,mean_RMS_w); \n'
  
  ##################################### pass wafer to Root ###################
  text_lsf += ' #pass wafers to object \n'
  text_lsf += ' select("::Root Element");  \n' 
  text_lsf += ' set("wafer_uniformity_thickness", wafer_uniformity_thickness);  \n'
  text_lsf += ' set("wafer_uniformity_width", wafer_uniformity_width);  \n'
  text_lsf += ' set("N",N);  \n'
  text_lsf += ' set("wafer_length",wafer_length);  \n'
  
  #################################### embed wafer selection script in Root ###################
  text_lsf += ' select("::Root Element");\n'
  text_lsf += ' set("setup script",'+ "'" +  ' \n'
  text_lsf += '  ######################## high resolution interpolation for dies ################# \n'
  text_lsf += '  MC_grid = 5e-6;  \n'   # [m], mesh grid
  text_lsf += '  die_span_x = %s; \n'  % 5e-3 # datadict["die_length_x"]  # [m]    GUI INPUT: die length X
  text_lsf += '  die_span_y = %s; \n'  % 5e-3 # datadict["die_length_y"]  # [m]    GUI INPUT: die length Y
  text_lsf += '  MC_resolution_x = die_span_x/MC_grid;  \n'
  text_lsf += '  MC_resolution_y = die_span_y/MC_grid;  \n'
  text_lsf += '  die_num_x = floor(wafer_length/die_span_x); \n'
  text_lsf += '  die_num_y = floor(wafer_length/die_span_y); \n'
  text_lsf += '  die_num_total = die_num_x*die_num_y; \n'
  text_lsf += '  x = linspace(-wafer_length/2,wafer_length/2,N); \n'
  text_lsf += '  y = linspace(-wafer_length/2,wafer_length/2,N); \n'
              # pick die for simulation, and do high resolution interpolation 
  text_lsf += '  j=selected_die; \n'
  text_lsf += '  die_min_x = -wafer_length/2+(j-1)*die_span_x -floor((j-1)/die_num_x)*wafer_length; \n'
  text_lsf += '  die_max_x = -wafer_length/2+j*die_span_x -floor((j-1)/die_num_x)*wafer_length; \n'
  text_lsf += '  die_min_y = wafer_length/2-ceil(j/die_num_y)*die_span_y; \n'
  text_lsf += '  die_max_y = wafer_length/2-(ceil(j/die_num_y)-1)*die_span_y; \n'
  text_lsf += '  x_die = linspace(die_min_x, die_max_x, MC_resolution_x); \n'
  text_lsf += '  y_die = linspace(die_min_y, die_max_y, MC_resolution_y); \n'
  text_lsf += '  die_xx = meshgridx(x_die,y_die) ;  \n'
  text_lsf += '  die_yy = meshgridy(x_die,y_die) ;  \n'
  text_lsf += '  MC_uniformity_thickness = interp(wafer_uniformity_thickness, x, y, x_die, y_die); # interpolation \n'
  text_lsf += '  MC_uniformity_width = interp(wafer_uniformity_width, x, y, x_die, y_die); # interpolation \n'
  ######################### pass die to object ####################################
  text_lsf += '  select("::Root Element::%s");  \n' % circuit_name
  text_lsf += '  set("MC_uniformity_thickness",MC_uniformity_thickness);  \n'
  text_lsf += '  set("MC_uniformity_width",MC_uniformity_width);  \n'
  text_lsf += '  set("MC_resolution_x",MC_resolution_x);  \n'
  text_lsf += '  set("MC_resolution_y",MC_resolution_y);  \n'
  text_lsf += '  set("MC_grid",MC_grid);  \n'
  text_lsf += '  set("MC_non_uniform",1);  \n'
  text_lsf += " '"+'); \n'
  
  text_lsf += ' for (ii=1;  ii<=n_die; ii=ii+1) {   \n'
  text_lsf += '  switchtodesign; \n'
  text_lsf += '  setnamed("ONA_1","peak analysis", "single");\n'
  text_lsf += '  select("::Root Element");  \n'
  text_lsf += '  set("selected_die",ii);  \n'
  text_lsf += '  run;\n'
  text_lsf += '  select("ONA_1");\n'
  text_lsf += '  T=getresult("ONA_1","input 1/mode 1/transmission");\n'
  text_lsf += '  wavelength = T.wavelength;\n'   
  
  for i in range(0, num_detectors):
    text_lsf += '  if (kk==1) { mc%s.addparameter("wavelength",wavelength);} \n' % (i+1) 
    text_lsf += '  mc%s.addattribute("run", getattribute( getresult("ONA_1", "input %s/mode 1/gain"), getattribute(getresult("ONA_1", "input %s/mode 1/gain")) ) );\n' % (i+1, i+1, i+1)
    text_lsf += '  Gain_Data_input%s(1:num_points, kk) = getattribute( getresult("ONA_1", "input %s/mode 1/gain"), getattribute(getresult("ONA_1", "input %s/mode 1/gain")) ); \n'  % (i+1, i+1, i+1)
    
#add simulation data to their corresponding datalists  
  if(params['histograms']['fsr']==True):
      text_lsf += '  fsr_select = getresult("ONA_1", "input 1/mode 1/peak/free spectral range");\n'
      text_lsf += '  fsr_dataset(1,kk) = real(fsr_select.getattribute(getattribute(fsr_select)));\n'

  if(params['histograms']['wavelength']==True):
      text_lsf += '  freq_dataset(1,kk) = getresult("ONA_1", "input 1/mode 1/peak/frequency");\n'

  if(params['histograms']['gain']==True):
      text_lsf += '  gain_select = getresult("ONA_1", "input 1/mode 1/peak/gain");\n'
      text_lsf += '  gain_dataset(1,kk) = real(gain_select.getattribute(getattribute(gain_select)));\n'

  text_lsf += '  switchtodesign; \n'
  text_lsf += '  kk = kk + 1;  \n'
  text_lsf += ' }\n'   # end for wafer iteration
  text_lsf += '}\n'  # end for die iteration
  text_lsf += '?"Spectrum data for each input can be found in the Script Workspace tab:";\n'    
  for i in range(0, num_detectors): 
      text_lsf += '?"Gain_Data_input%s"; \n' %(i+1)
  text_lsf += '?"Plot spectrums using script: plot(wavelength, Gain_Data_input#)";\n'  
  for i in range(0, num_detectors):
    text_lsf += 'visualize(mc%s);\n' % (i+1)
  
#### Display Histograms for the selected components
#FSR
  if(params['histograms']['fsr']==True):
      text_lsf += 'dataset = fsr_dataset*1e9;\n'  #select fsr dataset defined above
      text_lsf += 'bin_hist = max( [ 10, (max(dataset)-min(dataset)) / std(dataset) * 10 ]);\n' #define number of bins according to the number of data
      text_lsf += 'histc(dataset, bin_hist, "Free Spectral Range (nm)", "Count", "Histogram - FSR");\n' #generate histogram 
      text_lsf += 'legend("Mean: " + num2str(mean(dataset)) + ", Std Dev: " + num2str(std(dataset)));\n' #define plot legends
      
#wavelength
  if(params['histograms']['wavelength']==True):
      text_lsf += 'dataset = freq_dataset*1e9;\n'
      text_lsf += 'num_hist = max( [ 10, (max(dataset)-min(dataset)) / std(dataset) * 10 ]);\n'
      text_lsf += 'histc(dataset, bin_hist, "Wavelength (nm)", "Count", "Histogram - Peak wavelength");\n'
      text_lsf += 'legend("Mean: " + num2str(mean(dataset)) + ", Std Dev: " + num2str(std(dataset)));\n'

#Gain
  if(params['histograms']['gain']==True):
      text_lsf += 'dataset = gain_dataset;\n'
      text_lsf += 'num_hist = max( [ 10, (max(dataset)-min(dataset)) / std(dataset) * 10 ]);\n'
      text_lsf += 'histc(dataset, bin_hist, "Gain (dB)", "Count", "Histogram - Peak gain");\n'
      text_lsf += 'legend("Mean: " + num2str(mean(dataset)) + ", Std Dev: " + num2str(std(dataset)));\n'





  '''

  for i in range(1, num_detectors+1):
    if matlab_data_files:
      # convert simulation data into simple datasets:
      wavelenth_scale = 1e9
      text_lsf += 'temp = getresult("ONA_1", "input %s/mode 1/gain");\n' % i
      text_lsf += 't%s = matrixdataset("Simulation");\n' % i
      text_lsf += 't%s.addparameter("wavelength",temp.wavelength*%s);\n' % (i, wavelenth_scale)
      text_lsf += 't%s.addattribute("Simulation, Detector %s",getresultdata("ONA_1", "input %s/mode 1/gain"));\n' % (i,i, i)
    else:
      text_lsf += 't%s = getresult("ONA_1", "input %s/mode 1/gain");\n' % (i, i)
      
  # load measurement data files
  m_count=0
  if matlab_data_files:
    for m in matlab_data_files:
      if '.mat' in m:
        m_count += 1
        # INTERCONNECT can't deal with our measurement files... load and save data.
        from scipy.io import loadmat, savemat        # used to load MATLAB data files
        # *** todo, use DFT rules to determine which measurements we should load.
        PORT=2 # Which Fibre array port is the output connected to?
        matData = loadmat(m, squeeze_me=True, struct_as_record=False)
        wavelength = matData['scandata'].wavelength
        power = matData['scandata'].power[:,PORT-1]
        savemat(m, {'wavelength': wavelength, 'power': power})
        
        # INTERCONNECT load data
        head, tail = os.path.split(m)
        tail = tail.split('.mat')[0]
        text_lsf += 'matlabload("%s");\n' % m
        text_lsf += 'm%s = matrixdataset("Measurement");\n' % m_count
        text_lsf += 'm%s.addparameter("wavelength",wavelength*%s);\n'  % (m_count, wavelenth_scale)
        text_lsf += 'm%s.addattribute("Measured: %s",power);\n'  % (m_count, tail)
  
  text_lsf += 'visualize(t1'
  for i in range(2, num_detectors+1):
    text_lsf += ', t%s' % i
  for i in range(1, m_count+1):
    text_lsf += ', m%s' % i
  text_lsf += ');\n'
  
  '''
  
  file.write (text_lsf)
  file.close()
  
  if verbose:
    print(text_lsf)

  if simulate:
    # Run using Python integration:
    try: 
      from .. import _globals
      run_INTC()
      # Run using Python integration:
      lumapi = _globals.LUMAPI
      lumapi.evalScript(_globals.INTC, "cd ('" + tmp_folder + "');")
      lumapi.evalScript(_globals.INTC, "feval('"+ circuit_name + "');\n")
    except:
      from .. import scripts
      scripts.open_folder(tmp_folder)
      INTC_commandline(filename)
  else:
    from .. import scripts
    scripts.open_folder(tmp_folder)
    
  if verbose:
    print('Done Lumerical INTERCONNECT Monte Carlo circuit simulation.')

