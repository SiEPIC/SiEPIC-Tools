import pya


def registerMenuItems():
    import os
    from . import scripts, examples, lumerical, install
    import SiEPIC.__init__

    global ACTIONS
    count = 0
    menu = pya.Application.instance().main_window().menu()
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "files", "INTERCONNECT_icon.png")

    import sys
    if int(sys.version[0]) > 2 and sys.platform == 'darwin':
        extra = " Py3"
    else:
        extra = ''

    s1 = "siepic_menu"
    if not(menu.is_menu(s1)):
        menu.insert_menu("help_menu", s1, "SiEPIC %s" % SiEPIC.__init__.__version__ + extra)

    s2 = "waveguides"
    if not(menu.is_menu(s1 + "." + s2)):
        menu.insert_menu(s1 + ".end", s2, "Waveguides")

    s2 = "layout"
    if not(menu.is_menu(s1 + "." + s2)):
        menu.insert_menu(s1 + ".end", s2, "Layout")

    s2 = "exlayout"
    if not(menu.is_menu(s1 + "." + s2)):
        menu.insert_menu(s1 + ".end", s2, "Example Layouts")

    s2 = "verification"
    if not(menu.is_menu(s1 + "." + s2)):
        menu.insert_menu(s1 + ".end", s2, "Verification")

    s2 = "simulation"
    if not(menu.is_menu(s1 + "." + s2)):
        menu.insert_menu(s1 + ".end", s2, "Simulation")

    s2 = "measurements"
    if not(menu.is_menu(s1 + "." + s2)):
        menu.insert_menu(s1 + ".end", s2, "Measurement Data")

    if not(menu.is_menu("@toolbar.cir_sim")):
        ACTIONS.append(pya.Action())
        menu.insert_item("@toolbar.end", "cir_sim", ACTIONS[count])
    ACTIONS[count].title = "Circuit \nSimulation"
    ACTIONS[count].on_triggered(lumerical.interconnect.circuit_simulation)
    ACTIONS[count].icon = path
    count += 1

    if not(menu.is_menu("@toolbar.cir_sim.mc_sim")):
        ACTIONS.append(pya.Action())
        menu.insert_item("@toolbar.cir_sim.end", "mc_sim", ACTIONS[count])
    ACTIONS[count].title = "INTERCONNECT Monte Carlo Simulations"
    ACTIONS[count].on_triggered(lumerical.interconnect.circuit_simulation_monte_carlo)
    ACTIONS[count].icon = path
    count += 1

    if not(menu.is_menu("@toolbar.cir_sim.launch_lumerical")):
        ACTIONS.append(pya.Action())
        menu.insert_item("@toolbar.cir_sim.end", "launch_lumerical", ACTIONS[count])
    ACTIONS[count].title = "INTERCONNECT Circuit Simulation"
    ACTIONS[count].on_triggered(lumerical.interconnect.circuit_simulation)
    ACTIONS[count].icon = path
    count += 1

    if not(menu.is_menu("@toolbar.cir_sim.update_netlist")):
        ACTIONS.append(pya.Action())
        menu.insert_item("@toolbar.cir_sim.end", "update_netlist", ACTIONS[count])
    ACTIONS[count].title = "INTERCONNECT Update Netlist"
    ACTIONS[count].on_triggered(lumerical.interconnect.circuit_simulation_update_netlist)
    ACTIONS[count].icon = path


def registerKeyBindings():
    import os

    config = pya.Application.instance().get_config('key-bindings')
    if config == '':
        print('WARNING: get_config(key-bindings) returned null')
        mapping = dict()
    else:
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
    mapping['edit_menu.select_menu.select_all'] = "'Shift+Ctrl+A'"
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
    mapping['zoom_menu.select_current_cell'] = "'Shift+S'"  # Display > Show as new top
    mapping['zoom_menu.zoom_fit'] = "'F'"
    mapping['zoom_menu.zoom_fit_sel'] = "'Shift+F2'"
    mapping['zoom_menu.zoom_in'] = "'Return'"
    mapping['zoom_menu.zoom_out'] = "'Shift+Return'"

    # turn the hash back into a config string
    config = ''.join('{}:{};'.format(key, val) for key, val in sorted(mapping.items()))[:-1]
    pya.Application.instance().set_config('key-bindings', config)
    pya.Application.instance().set_config('edit-connect-angle-mode', 'ortho')
    pya.Application.instance().set_config('edit-inst-angle', '0')
    pya.Application.instance().set_config('edit-move-angle-mode', 'diagonal')
    pya.Application.instance().set_config('edit-snap-to-objects', 'true')
    pya.Application.instance().set_config('grid-micron', '0.01')
    pya.Application.instance().set_config('edit-top-level-selection', 'true')
    pya.Application.instance().set_config('inst-color', '#ffcdcd')
    pya.Application.instance().set_config('text-font', '3')
    pya.Application.instance().set_config('guiding-shape-line-width', '0')
    pya.Application.instance().set_config('rdb-marker-color', '#ff0000')
    pya.Application.instance().set_config('rdb-marker-line-width', '8')
#    pya.Application.instance().set_config('default-layer-properties', os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir, 'libraries', 'klayout_Layers_EBeam.lyp'))

    if pya.Application.instance().get_config('edit-mode') == 'false':
        pya.Application.instance().set_config('edit-mode', 'true')
        pya.MessageBox.warning(
            "Restart", "Please restart KLayout. SiEPIC settings have been applied.", pya.MessageBox.Ok)
