'''
    Contains the workhorse class-generating function GDSCell.
    This allows new classes for particular GDS files to inherit lots of the shared functionality.
'''

import pya
import os
from siepic_tools.utils.pcells import CellWithPosition


def get_lib_cell(layout, name, libname="EBeam"):
    cell = layout.cell(name)
    if cell is None:
        cell = layout.create_cell(name, libname)
    if cell is None:
        raise RuntimeError(f'cell {name} not found in {libname}')
    return cell


def GDSCell(cell_name, filename=None, from_library=None, gds_dir=None):
    '''
        Args:
            cell_name: cell within that file.
            filename: is the gds file name.
            from_library: library name, e.g. EBeam

        Returns:
            (class) a GDS_cell_base class that can be inherited
    '''

    assert gds_dir is not None

    class GDS_cell_base(CellWithPosition):
        """ Imports a gds file and places it."""

        def __init__(self, name=cell_name, params=None):
            super().__init__(name, params=params)

        def pcell(self, layout, cell=None, params=None):
            if cell is None:
                cell = layout.create_cell(self.name)

            cp = self.parse_param_args(params)
            origin, _, _ = CellWithPosition.origin_ex_ey(self, params, multiple_of_90=True)

            if from_library is not None:
                # Topcell must be same as filename
                gdscell = get_lib_cell(layout, cell_name, from_library)
            else:
                # BUG loading this file twice segfaults klayout
                layout2 = pya.Layout()
                layout2.read(os.path.join(gds_dir, filename))
                gdscell2 = layout2.cell(cell_name)
                gdscell = layout.create_cell(cell_name)
                gdscell.copy_tree(gdscell2)
                del gdscell2
                del layout2

            rot_DTrans = pya.DTrans.R0
            angle_multiple_90 = cp.angle_ex // 90
            rot_DTrans.rot = (angle_multiple_90) % 4

            cell.insert(
                pya.DCellInstArray(gdscell.cell_index(),
                                   pya.DTrans(rot_DTrans, origin)))

            return cell, {}

    return GDS_cell_base
