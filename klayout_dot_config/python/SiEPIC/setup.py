import pya

def registerMenuItems():
  from . import scripts, examples, lumerical, _globals, install
  
  menu = pya.Application.instance().main_window().menu()
  s1 = "siepic_menu"
  if not(menu.is_menu(s1)):
    menu.insert_menu("help_menu",s1,"SiEPIC") 

  s2 = "install"
  if not(menu.is_menu(s1 + "." + s2)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(install.install_dependencies)
    _globals.ACTIONS[-1].title = "Install SiEPIC Dependencies"
    menu.insert_item(s1 + ".end" , s2, _globals.ACTIONS[-1])

  s2 = "waveguides"
  if not(menu.is_menu(s1 + "." + s2)):
    menu.insert_menu(s1 + ".end" , s2, "Waveguides")
  
  s3 = "path_to_wg"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.waveguide_from_path)
    _globals.ACTIONS[-1].shortcut = 'W'
    _globals.ACTIONS[-1].title = "Path to Waveguide"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
  
  s3 = "wg_to_path"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.waveguide_to_path)
    _globals.ACTIONS[-1].shortcut = 'Shift+W'
    _globals.ACTIONS[-1].title = "Waveguide to Path"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
    
  s3 = "measure_wg"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.waveguide_length)
    _globals.ACTIONS[-1].shortcut = 'Alt+Shift+W'
    _globals.ACTIONS[-1].title = "Measure Waveguide Length"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
  
  s3 = "measure_wg_diff"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.waveguide_length_diff)
    _globals.ACTIONS[-1].shortcut = 'Shift+D'
    _globals.ACTIONS[-1].title = "Measure Waveguide Length Difference"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
    
  s3 = "wg_heal"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.waveguide_heal)
    _globals.ACTIONS[-1].shortcut = 'H'
    _globals.ACTIONS[-1].title = "Heal Waveguides"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
   
  s2 = "layout"
  if not(menu.is_menu(s1 + "." + s2)):
    menu.insert_menu(s1 + ".end", s2, "Layout")
  
  s3 = "auto_route"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.auto_route)
    _globals.ACTIONS[-1].title = "Automated A* Routing"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
    
  s3 = "snap_component"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.snap_component)
    _globals.ACTIONS[-1].shortcut = 'Shift+O'
    _globals.ACTIONS[-1].title = "Snap Selected Component to Nearest Pins"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])  
  
  s3 = "delete_top_cells"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.delete_top_cells)
    _globals.ACTIONS[-1].title = "Delete Extra Top Cells"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
  
  s2 = "exlayout"
  if not(menu.is_menu(s1 + "." + s2)):
    menu.insert_menu(s1 +".end", s2,"Example Layouts")
    
  s3 = "dbl_bus_ring_res"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(examples.dbl_bus_ring_res)
    _globals.ACTIONS[-1].title = "Double Bus Ring Resonator"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
  
  s3 = "mzi"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(examples.mzi)
    _globals.ACTIONS[-1].title = "12 TE Mach-Zehner Interferometers - Jaspreet Jhoja"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
  
  s2 = "verification"
  if not(menu.is_menu(s1 + "." + s2)):
    menu.insert_menu(s1 + ".end", s2, "Verification")
  
  s3 = "compute_area"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.calculate_area)
    _globals.ACTIONS[-1].title = "Compute Area of Silicon Layers"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
    
  s3 = "calibreDRC"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.calibreDRC)
    _globals.ACTIONS[-1].title = "Remote Calibre DRC"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
  
  s3 = "auto_coord_extract"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.auto_coord_extract)
    _globals.ACTIONS[-1].title = "Automated Coordinate Extraction"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
  
  s3 = "layout_check"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.layout_check)
    _globals.ACTIONS[-1].shortcut = 'V'
    _globals.ACTIONS[-1].title = "Verification: Graphical Layout Check"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
    
  s3 = "text_netlist_check"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(scripts.text_netlist_check)
    _globals.ACTIONS[-1].shortcut = 'N'
    _globals.ACTIONS[-1].title = "Verification: Text and Netlist Generation"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
  
  s2 = "simulation"
  if not(menu.is_menu(s1 + "." + s2)):
    menu.insert_menu(s1 + ".end", s2, "Simulation")
    
  s3 = "mc_sim"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(lumerical.interconnect.monte_carlo)
    _globals.ACTIONS[-1].title = "Monte Carlo Simulation"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
  
  s3 = "launch_lumerical"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(lumerical.interconnect.launch)
    _globals.ACTIONS[-1].title = "Launch Lumerical Interconnect"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])
    
  s3 = "update_netlist"
  if not(menu.is_menu(s1 + "." + s2 + "." + s3)):
    _globals.ACTIONS.append(pya.Action())
    _globals.ACTIONS[-1].on_triggered(lumerical.interconnect.update_netlist)
    _globals.ACTIONS[-1].title = "Update Netlist in Lumerical Inteconnect"
    menu.insert_item(s1 + "." + s2 + ".end" , s3, _globals.ACTIONS[-1])

def registerInterconnectToolbar():
  import os
  from .lumerical import interconnect
  from . import _globals
  
  menu = pya.Application.instance().main_window().menu()
  path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "files", "INTERCONNECT_icon.png")
  
  _globals.ACTIONS.append(pya.Action())
  _globals.ACTIONS[-1].title = "Circuit \nSimulation"
  _globals.ACTIONS[-1].on_triggered(interconnect.launch)
  _globals.ACTIONS[-1].icon = path
  menu.insert_item("@toolbar.end", "cir_sim", _globals.ACTIONS[-1])
  
  _globals.ACTIONS.append(pya.Action())
  _globals.ACTIONS[-1].title = "INTERCONNECT Monte Carlo Simulations"
  _globals.ACTIONS[-1].on_triggered(interconnect.monte_carlo)
  _globals.ACTIONS[-1].icon = path
  menu.insert_item("@toolbar.cir_sim.end", "mc_sim", _globals.ACTIONS[-1])
  
  _globals.ACTIONS.append(pya.Action())
  _globals.ACTIONS[-1].title = "INTERCONNECT Circuit Simulation"
  _globals.ACTIONS[-1].on_triggered(interconnect.launch)
  _globals.ACTIONS[-1].icon = path
  menu.insert_item("@toolbar.cir_sim.end", "launch_lumerical", _globals.ACTIONS[-1])
  
  _globals.ACTIONS.append(pya.Action())
  _globals.ACTIONS[-1].title = "INTERCONNECT Update Netlist"
  _globals.ACTIONS[-1].on_triggered(interconnect.update_netlist)
  _globals.ACTIONS[-1].icon = path
  menu.insert_item("@toolbar.cir_sim.end", "update_netlist", _globals.ACTIONS[-1])

def registerKeyBindings():
  import os

  config = pya.Application.instance().get_config('key-bindings')
  mapping = dict(item.split(":") for item in config.split(";"))
  
  mapping['edit_menu.clear_all_rulers'] = "'Ctrl+K'"
  mapping['edit_menu.copy'] = "'Ctrl+C'"
  mapping['edit_menu.cut'] = "'Ctrl+X'"
  mapping['edit_menu.paste'] = "'Ctrl+V'"
  mapping['edit_menu.redo'] = "'Ctrl+Y'"
  mapping['edit_menu.undo'] = "'Ctrl+Z'"
  mapping['edit_menu.delete'] = "'Del'"
  #  mapping['edit_menu.duplicate'] = "'Ctrl+B'"
  mapping['edit_menu.mode_menu.move'] = "'M'"
  mapping['edit_menu.mode_menu.ruler'] = "'R'"
  mapping['edit_menu.mode_menu.select'] = "'S'"
  mapping['edit_menu.mode_menu.box'] = "'B'"
  mapping['edit_menu.mode_menu.instance'] = "'I'"
  mapping['edit_menu.mode_menu.partial'] = "'L'"
  mapping['edit_menu.mode_menu.path'] = "'P'"
  mapping['edit_menu.mode_menu.polygon'] = "'G'"
  mapping['edit_menu.mode_menu.text'] = "'X'"
  mapping['edit_menu.select_menu.select_all'] = "'Ctrl+A'"
  mapping['edit_menu.show_properties'] = "'Q'"
  mapping['edit_menu.edit_options'] = "'E'"
  mapping['edit_menu.selection_menu.change_layer'] = "'Shift+L'"
  mapping['edit_menu.selection_menu.sel_flip_x'] = "'Shift+H'"
  mapping['edit_menu.selection_menu.sel_flip_y'] = "'Shift+V'"
  mapping['edit_menu.selection_menu.sel_move'] = "'Ctrl+M'"
  #  mapping['edit_menu.selection_menu.size'] = "'Z'"
  #  mapping['edit_menu.selection_menu.tap'] = "''"
  
  mapping['file_menu.new_layout'] = "'Ctrl+N'"
  mapping['file_menu.close'] = "'Ctrl+W'"
  mapping['file_menu.open_new_panel'] = "'Ctrl+O'"
  mapping['file_menu.open_same_panel'] = "'Ctrl+Shift+O'"
  mapping['file_menu.save'] = "'Ctrl+S'"
  mapping['file_menu.save_as'] = "'Ctrl+Shift+S'"
  mapping['file_menu.screenshot'] = "'F12'"
  #  mapping['file_menu.setup'] = "'F4'"
  
  mapping['macros_menu.macro_development'] = "'F5'"
  
  mapping['zoom_menu.max_hier'] = "'Shift+F'"
  mapping['zoom_menu.select_current_cell'] = "'Shift+S'" # Display > Show as new top
  mapping['zoom_menu.zoom_fit'] = "'F'"
  mapping['zoom_menu.zoom_fit_sel'] = "'Shift+F2'"
  mapping['zoom_menu.zoom_in'] = "'Return'"
  mapping['zoom_menu.zoom_out'] = "'Shift+Return'"
  
  # turn the hash back into a config string
  config = ''.join('{}:{};'.format(key, val) for key, val in sorted(mapping.items()))[:-1]
  pya.Application.instance().set_config('key-bindings', config)
  pya.Application.instance().set_config('edit-connect-angle-mode','ortho')
  pya.Application.instance().set_config('edit-inst-angle','0')
  pya.Application.instance().set_config('edit-move-angle-mode','diagonal')
  pya.Application.instance().set_config('edit-snap-to-objects','true')
  pya.Application.instance().set_config('grid-micron','0.01')
  pya.Application.instance().set_config('edit-top-level-selection','true')
  pya.Application.instance().set_config('inst-color','#ffcdcd')
  pya.Application.instance().set_config('text-font','3')
  pya.Application.instance().set_config('guiding-shape-line-width','0')
  pya.Application.instance().set_config('rdb-marker-color','#ff0000')
  pya.Application.instance().set_config('rdb-marker-line-width','8')
  pya.Application.instance().set_config('default-layer-properties', os.path.join(pya.Application.instance().application_data_path(),'salt','siepic_tools_dev','libraries','klayout_Layers_EBeam.lyp'))