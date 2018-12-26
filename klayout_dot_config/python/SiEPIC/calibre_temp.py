
def calibreDRC(params = None, cell = None):

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

  from . import _globals

  from .utils import load_Calibre
  CALIBRE = load_Calibre()
  if not CALIBRE:
    pya.MessageBox.warning("Missing CALIBRE.xml", "Missing CALIBRE.xml. This configuration file should be located in the tech folder for the active technology.",  pya.MessageBox.Ok)
    return    
  
  if 1:
    lv.transaction("calibre drc")
    
    import time
    progress = pya.RelativeProgress("Calibre DRC", 5)
    progress.format = "Saving Layout to Temporary File"
    progress.set(1, True)
    time.sleep(1)
    pya.Application.instance().main_window().repaint()

    # Local temp folder:
    from . import _globals
    import os
    local_path = _globals.TEMP_FOLDER
    local_file = os.path.basename(lv.active_cellview().filename())
    local_pathfile = os.path.join(local_path, local_file)
    
    remote_path = "/tmp/%s" % layout_basefilename
    remote_pathfile = remote_path + "/" + layout_filename
    print(" Creating remote temp folder: ")
    cmd = '/usr/bin/ssh %s "%s"' % (ssh_server, "mkdir -p %s" % remote_path ) 
    print(cmd)
    out = commands.getstatusoutput( cmd )
    print (out)

    results_file = layout_basefilename + "_litho.gds"
    results_pathfile = os.path.dirname(layout_path) + "/" + results_file

    cal_script = "#!/bin/tcsh \n"
    cal_script += "source %s \n" % remote_calibre_env_script
    cal_script += "cd %s\n" % remote_path


    
    results_file = os.path.basename(local_pathfile) + ".rve"
    results_pathfile = os.path.join(os.path.dirname(local_pathfile), results_file)
    tmp_ly = ly.dup()
    [cell.flatten(True) for cell in tmp_ly.each_cell()]
    opts = pya.SaveLayoutOptions()
    opts.format = "GDS2"
    tmp_ly.write(local_pathfile, opts)
    
    with codecs.open(os.path.join(local_path, 'run_calibre'), 'w', encoding="utf-8") as file:
      cal_script  = '#!/bin/tcsh \n'
      cal_script += 'source %s \n' % CALIBRE['remote_calibre_script']
      cal_script += '%s \n' % CALIBRE['remote_additional_commands']
      cal_script += '$MGC_HOME/bin/calibre -drc -hier -turbo -nowait drc.cal \n'
      file.write(cal_script)

    with codecs.open(os.path.join(local_path, 'drc.cal'), 'w', encoding="utf-8") as file:
      cal_deck  = 'LAYOUT PATH  "%s"\n' % os.path.basename(local_pathfile)
      cal_deck += 'LAYOUT PRIMARY "%s"\n' % cell.name
      cal_deck += 'LAYOUT SYSTEM GDSII\n'
      cal_deck += 'DRC RESULTS DATABASE "drc.rve" ASCII\n'
      cal_deck += 'DRC MAXIMUM RESULTS ALL\n'
      cal_deck += 'DRC MAXIMUM VERTEX 4096\n'
      cal_deck += 'DRC CELL NAME YES CELL SPACE XFORM\n'
      cal_deck += 'VIRTUAL CONNECT COLON NO\n'
      cal_deck += 'VIRTUAL CONNECT REPORT NO\n'
      cal_deck += 'DRC ICSTATION YES\n'
      cal_deck += 'INCLUDE "%s"\n' % CALIBRE['remote_calibre_rule_deck_main_file']
      file.write(cal_deck)

    import platform
    if 'Darwin' in platform.version():
      # OSX
      print("Uploading layout and Calibre scripts: ")
      cmd = 'cd %s; ' % pipes.quote(os.path.dirname(layout_filename))
      cmd += '/usr/bin/scp %s %s %s %s:%s' % (os.path.basename(layout_filename), "/tmp/run_calibre", "/tmp/drc.cal", ssh_server, remote_path) 
      print(cmd)
      out = commands.getstatusoutput( cmd )
      print (out)
           
      print("Checking layout for errors: ")
      cmd = '/usr/bin/ssh %s "%s"' % (ssh_server, "cd " + remote_path +"; source run_calibre;" ) 
      print(cmd)
      out = commands.getstatusoutput( cmd )
      print (out)
      
      print("Downloading results file: ")
      cmd = '/usr/bin/scp %s:%s %s' % ( ssh_server, remote_path + "/drc.rve", pipes.quote(results_pathfile) ) 
      print(cmd)
      out = commands.getstatusoutput( cmd )
      print (out)
      
      
       
    version = sys.version
    if version.find("2.") > -1:
      import commands

      progress.set(2, True)
      progress.format = "Uploading Layout and Scripts"
      pya.Application.instance().main_window().repaint()
      
      out = cmd('cd "%s" && scp -i C:/Users/bpoul/.ssh/drc -P%s "%s" %s:%s' % (os.path.dirname(local_pathfile), params['port'], os.path.basename(local_pathfile), params['url'], remote_path))
      out = cmd('cd "%s" && scp -i C:/Users/bpoul/.ssh/drc -P%s %s %s:%s' % (local_path, params['port'], 'run_calibre', params['url'], remote_path))
      out = cmd('cd "%s" && scp -i C:/Users/bpoul/.ssh/drc -P%s %s %s:%s' % (local_path, params['port'], 'drc.cal', params['url'], remote_path))

      progress.set(3, True)
      progress.format = "Checking Layout for Errors"
      pya.Application.instance().main_window().repaint()
    
      out = cmd('ssh -i C:/Users/bpoul/.ssh/drc -p%s %s "%s"' % (params['port'], params['url'], "cd " + remote_path +" && source run_calibre"))

      progress.set(4, True)
      progress.format = "Downloading Results"
      pya.Application.instance().main_window().repaint()
      
      out = cmd('cd "%s" && scp -i C:/Users/bpoul/.ssh/drc -P%s %s:%s %s' % (os.path.dirname(local_pathfile), params['port'], params['url'], remote_path + "/drc.rve", results_file))

      progress.set(5, True)
      progress.format = "Finishing"
      pya.Application.instance().main_window().repaint()
      
    elif version.find("3.") > -1:
      import subprocess
      cmd = subprocess.check_output

      progress.format = "Uploading Layout and Scripts"      
      progress.set(2, True)
      pya.Application.instance().main_window().repaint()
      
      out = cmd('cd "%s" && scp -i %s -P%s "%s" %s:%s' % (os.path.dirname(local_pathfile), params['identity'], params['port'], os.path.basename(local_pathfile), params['url'], remote_path), shell=True)
      out = cmd('cd "%s" && scp -i %s -P%s %s %s:%s' % (local_path, params['identity'], params['port'], 'run_calibre', params['url'], remote_path), shell=True)
      out = cmd('cd "%s" && scp -i %s -P%s %s %s:%s' % (local_path, params['identity'], params['port'], 'drc.cal', params['url'], remote_path), shell=True)

      progress.format = "Checking Layout for Errors"
      progress.set(3, True)
      pya.Application.instance().main_window().repaint()

      out = cmd('ssh -i %s -p%s %s "%s"' % (params['identity'], params['port'], params['url'], "cd " + remote_path +" && source run_calibre"), shell=True)
      
      progress.format = "Downloading Results"
      progress.set(4, True)
      pya.Application.instance().main_window().repaint()
      
      out = cmd('cd "%s" && scp -i %s -P%s %s:%s %s' % (os.path.dirname(local_pathfile), params['identity'], params['port'], params['url'], remote_path + "/drc.rve", results_file), shell=True)

      progress.format = "Finishing"
      progress.set(5, True)
      
    progress._destroy()
    if os.path.exists(results_pathfile):
      rdb_i = lv.create_rdb("Calibre Verification")
      rdb = lv.rdb(rdb_i)
      rdb.load (results_pathfile)
      rdb.top_cell_name = cell.name
      rdb_cell = rdb.create_cell(cell.name)
      lv.show_rdb(rdb_i, lv.active_cellview().cell_index)
    else:
      pya.MessageBox.warning("Errors", "Something failed during the server Calibre DRC check.",  pya.MessageBox.Ok)

    pya.Application.instance().main_window().update()
    lv.commit()
    