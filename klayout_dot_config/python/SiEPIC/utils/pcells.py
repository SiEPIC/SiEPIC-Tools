import pya
from SiEPIC.extend import to_dtype


class objectview(object):
    """ Basically allows us to access dictionary values as dict.x
        rather than dict['x']
    """

    def __init__(self, d):
        self.__dict__ = d


def place_cell(parent_cell, pcell, ports_dict, placement_origin, relative_to=None, transform_into=False):
    """ Places an pya cell and return ports with updated positions
    Args:
        parent_cell: cell to place into
        pcell, ports_dict: result of IMECell.pcell call
        placement_origin: pya.Point object to be used as origin
        relative_to: port name
            the cell is placed so that the port is located at placement_origin
        transform_into:
            if used with relative_into, transform the cell's coordinate system
            so that its origin is in the given port.

    Returns:
        ports(dict): key:port.name, value: geometry.Port with positions relative to parent_cell's origin
    """
    offset = pya.DVector(0, 0)
    port_offset = placement_origin
    if relative_to is not None:
        offset = ports_dict[relative_to].position
        port_offset = placement_origin - offset
        if transform_into:
            # print(type(pcell))
            offset_transform = pya.DTrans(pya.DTrans.R0, -offset)
            for instance in pcell.each_inst():
                instance.transform(offset_transform)
            pcell.transform_into(offset_transform)
        else:
            placement_origin = placement_origin - offset

    transformation = pya.DTrans(pya.Trans.R0, placement_origin)
    instance = pya.DCellInstArray(pcell.cell_index(), transformation)
    parent_cell.insert(instance)
    for port in ports_dict.values():
        port.position += port_offset

    return ports_dict


def port_to_pin_helper(ports_list, cell, layerPinRec):
    # Create the pins, as short paths:
    from SiEPIC._globals import PIN_LENGTH
    dbu = cell.layout().dbu

    for port in ports_list:
        if port.name.startswith("el"):
            pin_length = port.width
        else:
            pin_length = to_dtype(PIN_LENGTH, dbu)

        port_position_i = port.position.to_itype(dbu)
        cell.shapes(layerPinRec).insert(
            pya.DPath([port.position - 0.5 * pin_length * port.direction,
                       port.position + 0.5 * pin_length * port.direction], port.width).to_itype(dbu))
        cell.shapes(layerPinRec).insert(pya.Text(port.name, pya.Trans(
            pya.Trans.R0, port_position_i.x, port_position_i.y))).text_size = 2 / dbu


class KLayoutPCell(object):
    """ Main layout class for instantiating PDK-compatible cells """
    TypeBoolean = pya.PCellDeclarationHelper.TypeBoolean
    TypeDouble = pya.PCellDeclarationHelper.TypeDouble
    TypeInt = pya.PCellDeclarationHelper.TypeInt
    TypeLayer = pya.PCellDeclarationHelper.TypeLayer
    TypeList = pya.PCellDeclarationHelper.TypeList
    TypeNone = pya.PCellDeclarationHelper.TypeNone
    TypeShape = pya.PCellDeclarationHelper.TypeShape
    TypeString = pya.PCellDeclarationHelper.TypeString

    # helpful variable TECHNOLOGY
    # example: TECHNOLOGY = get_technology_by_name('EBeam')
    TECHNOLOGY = None

    def __init__(self, name="Unnamed Cell", params=None):
        ''' params can be a dict or an objectview '''
        self.name = name
        self.param_definition = dict()
        self.initialize_default_params()
        self.default_params = {name: value["default"]
                               for name, value in self.param_definition.items()}
        if params is not None:
            if isinstance(params, objectview):
                for k, v in params.__dict__.items():
                    if not k.endswith('__'):
                        self.default_params[k] = v
            else:
                self.default_params.update(params)

    def to_pya(parent_self):
        param_definition = parent_self.param_definition
        pcell = parent_self.pcell

        class PCell(pya.PCellDeclarationHelper):

            def __init__(self):
                super(PCell, self).__init__()
                # declare the parameters
                for name, param in param_definition.items():
                    try:
                        # Warning: param is mutable!
                        param_type = param["type"]
                        description = param["description"]
                        default = param["default"]
                        optional_params = {option: value for option, value in param.items()
                                           if value is not None and option not in ('type', 'description', 'default')}
                        self.param(name, param_type, description,
                                   default=default, **optional_params)
                    except Exception as e:
                        print("Invalid params in {}: {}".format(
                            parent_self.__class__.__name__, param))

            def display_text_impl(self):
                # This text shows when you choose not to see the full hierarchy
                # Currently printing all non-default parameters of pcell.
                text = "{}:\n".format(parent_self.__class__.__name__)
                for param, value in zip(self.get_parameters(), self.get_values()):
                    params_to_show = []
                    if value != param_definition[param.name]["default"]:
                        params_to_show.append("{}:{}".format(param.name, value))
                    text += ", ".join(params_to_show)
                return text

            def can_create_from_shape_impl(self):
                return False

            def produce_impl(self):
                params = {k: getattr(self, k) for k in param_definition.keys()}
                pcell(self.layout, self.cell, params)

        return PCell()

    def define_param(self, name, p_type, description, default, unit=None,
                     readonly=None, choices=None):
        self.param_definition[name] = {"type": p_type,
                                       "description": description,
                                       "default": default,
                                       "unit": unit,
                                       "readonly": readonly,
                                       "choices": choices
                                       }
        return self.param_definition[name]

    def initialize_default_params(self):
        """ Initializes a list of default parameters using define_param"""
        raise NotImplementedError()

    def parse_param_args(self, params):
        ''' returns a *new* parameter dictionary with defaults changed by params

            Returns:
                object: objectview of full parameter structure
                access with cp.name instead of cp['name']
        '''
        cell_params = dict(self.default_params)
        if params is not None:
            cell_params.update(params)
        return objectview(cell_params)

    def pcell(self, layout, cell=None, params=None):
        """ Draws cell contents into cell and return it.

        Args:
            layout (pya.Layout): layout object to attach cell on
            cell (pya.Cell): cell to draw on. If None, new is created
            params (dict): params to be overwritten from default.

        Returns:
            cell
            ports: dict of Port objects
        """
        raise NotImplementedError()

    def place_cell(self, parent_cell, origin, params=None, relative_to=None, transform_into=False):
        """ Places this cell and return ports
        Args:
            parent_cell: cell to place into
            ime_cell: IMECell to be placed
            placement_origin: pya.Point object to be used as origin
            relative_to: port name
                the cell is placed so that the port is located at placement_origin
            transform_into:
                if used with relative_into, transform the cell's coordinate system
                so that its origin is in the given port.

        Returns:
            ports(dict): key:port.name, value: geometry.Port with positions relative to parent_cell's origin
        """
        layout = parent_cell.layout()
        pcell, ports = self.pcell(layout, params=params)
        return place_cell(parent_cell, pcell, ports, origin, relative_to=relative_to, transform_into=transform_into)


# CACHE TOOLS

import os
from hashlib import sha256
import inspect
import pickle

layer_map_dict = dict()
debug = False
cache_dir = os.path.join(os.getcwd(), 'cache')


def cache_cell(cls, cache_dir=cache_dir):
    """ Caches results of pcell call to save build time.

    Use as a decorator:

        @cache_cell
        class MyCell(KLayoutPCell):
            pass
    """
    activated = True
    if activated:
        # decorate pcell
        def cache_decorator(pcell):
            def wrapper_pcell(self, layout, cell=None, params=None):
                global layer_map_dict
                try:
                    layer_map_dict[layout]
                except KeyError:
                    layer_map_dict[layout] = pya.LayerMap()
                if cell is None:
                    # copy source code of class and all its ancestors
                    source_code = "".join(
                        [inspect.getsource(klass) for klass in self.__class__.__mro__ if issubclass(klass, KLayoutPCell)])

                    # Default params before instantiation
                    original_default_params = {name: value["default"]
                                               for name, value in self.param_definition.items()}

                    # Updated params after instantiation and placement
                    # (could be bigger than the original default)
                    if params is not None:
                        default_params = dict(self.default_params, **params)
                    else:
                        default_params = self.default_params

                    # Differential parameters (non-default)
                    diff_params = {}
                    for name, value in original_default_params.items():
                        if default_params[name] != value:
                            diff_params[name] = default_params[name]

                    long_hash_pcell = sha256((source_code +
                                              str(diff_params) +
                                              self.name).encode()).hexdigest()
                    short_hash_pcell = long_hash_pcell[0:7]
                    cache_fname = f'cache_{self.__class__.__qualname__}_{short_hash_pcell}'
                    # if short_hash_pcell in cell_cache.keys():  # already loaded
                    #     print(f"Preloaded {self.__class__.__name__}: {diff_params}")
                    #     cached_cell, ports_bytecode, cellname = cell_cache[short_hash_pcell]
                    #     ports = pickle.loads(ports_bytecode)
                    #     # print('read:', cell_index, ports, cellname)
                    #     newcell = layout.create_cell(cellname)
                    #     newcell.copy_tree(cached_cell)
                    #     # newcell.insert(pya.DCellInstArray(cell.cell_index(),
                    #     #                                   pya.DTrans(pya.Trans.R0, pya.DPoint(0, 0))))
                    #     return newcell, deepcopy(ports)

                    def read_layout(layout, gds_filename):
                        global layer_map_dict
                        load_options = pya.LoadLayoutOptions()
                        load_options.text_enabled = True
                        load_options.set_layer_map(layer_map_dict[layout], True)

                        # store and take away the cell names of all cells read so far
                        # (by setting the cell name to "" the cells basically become invisible for
                        # the following read)
                        # take out the pcells
                        cell_list = [cell for cell in layout.each_cell()]
                        cell_indices = {cell.name: cell.cell_index() for cell in cell_list}
                        for i in cell_indices.values():
                            layout.rename_cell(i, "")
                        # pdb.set_trace()
                        # if 'cache_Ndoped_Ring_Filter' in gds_filename:
                        #     pdb.set_trace()
                        lmap = layout.read(gds_filename, load_options)
                        # in the new layout, get all cells with starting names cache_
                        cell_names2 = [(cell.cell_index(), cell.name)
                                       for cell in layout.each_cell()]
                        # pdb.set_trace()
                        # make those cells point to older cells
                        prune_cells_indices = []
                        for i_duplicate, name_cached_cell in cell_names2:
                            if name_cached_cell in cell_indices.keys():
                                if name_cached_cell.startswith('cache_'):
                                    for parent_inst_array in layout.cell(i_duplicate).each_parent_inst():
                                        cell_instance = parent_inst_array.child_inst()
                                        cell_instance.cell = layout.cell(
                                            cell_indices[name_cached_cell])
                                    prune_cells_indices.append(i_duplicate)
                                else:
                                    # print('RENAME', name_cached_cell)
                                    k = 1
                                    while (name_cached_cell + f"_{k}") in cell_indices.keys():
                                        k += 1
                                    layout.rename_cell(i_duplicate, name_cached_cell + f"_{k}")

                        for i_pruned in prune_cells_indices:
                            # print('deleting cell', layout.cell(i_pruned).name)
                            layout.prune_cell(i_pruned, -1)

                        # every conflict should have been caught above
                        for name, i in cell_indices.items():
                            layout.rename_cell(i, name)

                        layer_map_dict[layout] = lmap
                        return lmap

                    cache_fname_gds = cache_fname + '.gds'
                    cache_fname_pkl = cache_fname + '.klayout.pkl'

                    os.makedirs(cache_dir, mode=0o775, exist_ok=True)

                    cache_fpath_gds = os.path.join(cache_dir, cache_fname_gds)
                    cache_fpath_pkl = os.path.join(cache_dir, cache_fname_pkl)
                    if os.path.isfile(cache_fpath_gds) and os.path.isfile(cache_fpath_pkl):
                        with open(cache_fpath_pkl, 'rb') as file:
                            ports, read_short_hash_pcell, cellname = pickle.load(file)
                        if debug:
                            print(f"Reading from cache: {cache_fname}: {diff_params}, {cellname}")
                        else:
                            print('r', end='', flush=True)
                        if not layout.has_cell(cache_fname):
                            read_layout(layout, cache_fpath_gds)
                        retrieved_cell = layout.cell(cache_fname)
                        cell = layout.create_cell(cellname)
                        cell.insert(pya.DCellInstArray(retrieved_cell.cell_index(),
                                                       pya.DTrans(pya.Trans.R0, pya.DPoint(0, 0))))
                        # cell.move_tree(retrieved_cell)
                    else:
                        if layout.has_cell(cache_fname):
                            print(f"WARNING: {cache_fname_gds} does not exist but {cache_fname} is in layout.")

                        # populating .gds and .pkl
                        empty_layout = pya.Layout()
                        compiled_cell, ports = pcell(
                            self, empty_layout, cell=None, params=params)
                        if debug:
                            print(f"Writing to cache: {cache_fname}: {diff_params}, {compiled_cell.name}")
                        else:
                            print('w', end='', flush=True)
                        cellname, compiled_cell.name = compiled_cell.name, cache_fname
                        compiled_cell.write(cache_fpath_gds)
                        with open(cache_fpath_pkl, 'wb') as file:
                            pickle.dump((ports, short_hash_pcell, cellname), file)
                        read_layout(layout, cache_fpath_gds)

                        retrieved_cell = layout.cell(cache_fname)
                        cell = layout.create_cell(cellname)
                        cell.insert(pya.DCellInstArray(retrieved_cell.cell_index(),
                                                       pya.DTrans(pya.Trans.R0, pya.DPoint(0, 0))))

                else:
                    cell, ports = pcell(self, layout, cell=cell, params=params)
                return cell, ports
            return wrapper_pcell
        setattr(cls, 'pcell', cache_decorator(cls.__dict__['pcell']))
    return cls
