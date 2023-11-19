"""Send Layout (GDS, OAS) or Verification results (.lyrdb) to Klayout. Requires klive installed in KLayout."""

# adapted from gdsfactory.

import json
import os
import pathlib
import socket
from pathlib import Path



def show(
    gds_filename: str,
    lyrdb_filename = None,
    keep_position: bool = True,
    technology: str = '',
    port: int = 8082,
    delete: bool = False,
) -> None:
    """Show GDS or Layout ReportDatabase (.lyrdb) in klayout.

    Args:
        gds_filename: layout to show.
        lyrdb_filename: to open in KLayout > Tools > Marker Browser.
        keep_position: keep position and active layers.
        technology: Name of the KLayout technology to use.
        port: klayout server port.
        delete: deletes file.
    """
    if not os.path.isfile(gds_filename):
        raise ValueError(f"{gds_filename} does not exist")

    gds_filename = pathlib.Path(gds_filename)

    data = {
        "gds": os.path.abspath(gds_filename),
        "keep_position": keep_position,
        "technology": technology,
    }

    if lyrdb_filename:
        if not os.path.isfile(lyrdb_filename):
            raise ValueError(f"{lyrdb_filename} does not exist")

        lyrdb_filename = pathlib.Path(lyrdb_filename)
        data['lyrdb'] = os.path.abspath(lyrdb_filename)

    data_string = json.dumps(data)
    try:
        conn = socket.create_connection(("127.0.0.1", port), timeout=0.5)
        data = data_string + "\n"
        enc_data = data.encode()
        conn.sendall(enc_data)
        conn.settimeout(5)
    except OSError:
        print(
            "Could not connect to klive server. Is klayout open and klive plugin installed?"
        )
    else:
        msg = ""
        try:
            msg = conn.recv(1024).decode("utf-8")
            print(f"Message from klive: {msg}")
        except OSError:
            print("klive didn't send data, closing")
        finally:
            conn.close()

    if delete:
        Path(gds_filename).unlink()
        if lyrdb_filename:
            Path(lyrdb_filename).unlink()
    

if __name__ == "__main__":
    import sys, pya
    if 'Application' in dir(pya):
        print("this test case should be run outside of KLayout's IDE, e.g., VSCode.")
        exit(0)
    
    # Load SiEPIC tools from GitHub
    path_GitHub = '/Users/lukasc/Documents/GitHub/'
    if 'SiEPIC' not in sys.modules:
        print('loading SiEPIC from GitHub')
        path_siepic = os.path.join(path_GitHub, 'SiEPIC-Tools/klayout_dot_config/python')
        if not path_siepic in sys.path:
            sys.path.insert(0, path_siepic)
        import SiEPIC
    from SiEPIC._globals import Python_Env
    print('KLayout running in mode: %s' % Python_Env)
    from SiEPIC.utils.layout import new_layout, floorplan
    from SiEPIC.scripts import load_klayout_technology, instantiate_all_library_cells, zoom_out, export_layout

    # Load EBeam PDK from GitHub
    path_module = os.path.join(path_GitHub, 'SiEPIC_EBeam_PDK/klayout')
    path_lyt_file = os.path.join(path_GitHub, 'SiEPIC_EBeam_PDK/klayout/EBeam/EBeam.lyt')
    tech = load_klayout_technology('EBeam', path_module, path_lyt_file)

    from SiEPIC.utils.layout import new_layout, floorplan
    tech_name = 'EBeam'
    topcell, ly = new_layout(tech_name, 'top', GUI=False, overwrite = True)
    cell = topcell.layout().create_cell("ebeam_y_1550",tech_name)
    t = pya.Trans(pya.Trans.R0, 0, 0)
    topcell.insert(pya.CellInstArray(cell.cell_index(), t))

    path = os.path.dirname(os.path.realpath(__file__))
    filename = 'klive'
    file_out = export_layout(topcell, path, filename, format='oas')
    
    from SiEPIC.verification import layout_check

    file_lyrdb = os.path.join(path,filename+'.lyrdb')
    layout_check(cell = topcell, verbose=False, file_rdb=file_lyrdb)
    show(file_out,lyrdb_filename=file_lyrdb)
    
    if 0:
        import gdsfactory as gf

        c = gf.components.mzi()
        #c = gf.components.straight(length=10)
        gdspath = c.write_gds()
        show(gdspath)
