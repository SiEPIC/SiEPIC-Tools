# Required packages
from SiEPIC.install import install
if not install('defusedxml'):
    pya.MessageBox.warning(
    "Missing package", "The OPICS circuit simulator does not function without the package 'defusedxml'.",  pya.MessageBox.Ok)    



from typing import Any, Dict, List, Tuple
import cmath as cm
import time
import re
import itertools
import inspect
from copy import deepcopy
import numpy as np
from numpy import ndarray
from pathlib import PosixPath
from defusedxml.ElementTree import parse


def fromSI(value: str) -> float:
    """converts from SI unit values to metric

    Args:
        value (str): a value in SI units, e.g. 1.3u

    Returns:
        float: the value in metric units.
    """
    return float(value.replace("u", "e-6"))


def universal_sparam_filereader(
    nports: int, sfilename: str, sfiledir: PosixPath, format_type: str = "auto"
) -> Tuple[ndarray, ndarray]:
    """
    Function to automatically detect the sparameter file format and use appropriate method to delimit and format sparam data

    This function is a unified version of sparameter reader function defined in https://github.com/BYUCamachoLab/simphony

    Args:
        nports: Number of ports
        sfilename: XML look-up-table filename
        sfiledir: Path to the directory containing the XML file
        format_type: Format type. For more information: https://support.lumerical.com/hc/en-us/articles/360036618513-S-parameter-file-formats
    """
    numports = nports
    filename = sfiledir / sfilename

    if format_type == "auto":
        try:
            # print("try A")
            result = universal_sparam_filereader(nports, sfilename, sfiledir, "A")
            return result
        except Exception:
            try:
                # print("try B")
                result = universal_sparam_filereader(nports, sfilename, sfiledir, "B")
                return result
            except Exception:
                # print("try C")
                result = universal_sparam_filereader(nports, sfilename, sfiledir, "C")
                return result

    elif format_type == "A":
        """
        dc_halfring_te_1550
        Returns the s-parameters across some frequency range for the Sparam fileformat A

        input:
        ["port 1",""]
        ["port 2",""]
        ["port 3",""]
        ["port 4",""]
        ("port 1","mode 1",1,"port 1",1,"transmission")
        (101, 3)

        output:
        [frequency, s-parameters]
        """
        F = []
        S = []
        with open(filename, "r") as fid:
            for i in range(5):
                line = fid.readline()
            line = fid.readline()
            numrows = int(tuple(line[1:-2].split(","))[0])
            S = np.zeros((numrows, numports, numports), dtype="complex128")
            r = m = n = 0
            for line in fid:
                if line[0] == "(":
                    continue
                data = line.split()
                data = list(map(float, data))
                if m == 0 and n == 0:
                    F.append(data[0])
                S[r, m, n] = data[1] * np.exp(1j * data[2])
                r += 1
                if r == numrows:
                    r = 0
                    m += 1
                    if m == numports:
                        m = 0
                        n += 1
                        if n == numports:
                            break
        return (np.array(F), S)

    elif format_type == "B":
        """
        ebeam_bdc_te1550, nanotaper, ebeam_y_1550

        Returns the s-parameters across some frequency range for the Sparam fileformat A
        input:
        ('port 1','TE',1,'port 1',1,'transmission')
        (51,3)

        output:
        [frequency, s-parameters]
        """
        F = []
        S = []
        with open(filename, "r") as fid:
            line = fid.readline()
            line = fid.readline()
            numrows = int(tuple(line[1:-2].split(","))[0])
            S = np.zeros((numrows, numports, numports), dtype="complex128")
            r = m = n = 0
            for line in fid:
                if line[0] == "(":
                    continue
                data = line.split()
                data = list(map(float, data))
                if m == 0 and n == 0:
                    F.append(data[0])
                S[r, m, n] = data[1] * np.exp(1j * data[2])
                r += 1
                if r == numrows:
                    r = 0
                    m += 1
                    if m == numports:
                        m = 0
                        n += 1
                        if n == numports:
                            break
        return (np.array(F), S)

    elif format_type == "C":
        """
        ebeam_gc_te1550

        Returns the s-parameters across some frequency range for the Sparam fileformat A
        input:
        columns with space delimiter

        output:
        [frequency, s-parameters]
        """
        with open(filename) as fid:
            # grating coupler compact models have 100 points for each s-matrix index
            arrlen = 100

            lines = fid.readlines()
            F = np.zeros(arrlen)
            S = np.zeros((arrlen, 2, 2), "complex128")
            for i in range(0, arrlen):
                words = lines[i].split()
                F[i] = float(words[0])
                S[i, 0, 0] = cm.rect(float(words[1]), float(words[2]))
                S[i, 0, 1] = cm.rect(float(words[3]), float(words[4]))
                S[i, 1, 0] = cm.rect(float(words[5]), float(words[6]))
                S[i, 1, 1] = cm.rect(float(words[7]), float(words[8]))
            F = F[::-1]
            S = S[::-1, :, :]
        return (np.array(F), S)


def LUT_reader(filedir: PosixPath, lutfilename: str, lutdata: List[List[str]]):
    """
    Reads look up table data.

    Args:
        filedir: Directory of the XML look-up-table file.
        lutfilename: Look-up-table filename.
        lutdata: Look-up-table arguments.
    """
    xml = parse(filedir / lutfilename)
    root = xml.getroot()

    for node in root.iter("association"):
        sample = [[each.attrib["name"], each.text] for each in node.iter("value")]
        if sorted(sample[0:-1]) == sorted(lutdata):
            break
    sparam_file = sample[-1][1].split(";")
    return (sparam_file, xml, node)


def LUT_processor(
    filedir: PosixPath,
    lutfilename: str,
    lutdata: List[List[str]],
    nports: int,
    sparam_attr: str,
    verbose: bool = False,
) -> Tuple[Tuple[ndarray, ndarray], str]:
    """process look up table data"""
    start = time.time()
    sparam_file, xml, node = LUT_reader(filedir, lutfilename, lutdata)

    # read data
    if ".npz" in sparam_file[0] or ".npz" in sparam_file[-1]:
        npzfile = [each for each in sparam_file if ".npz" in each][0]
        tempdata = np.load(filedir / npzfile)
        sdata = (tempdata["f"], tempdata["s"])
        npz_file = npzfile

    else:
        if verbose:
            print("numpy datafile not found. reading sparam file instead..")

        sdata = universal_sparam_filereader(nports, sparam_file[-1], filedir, "auto")
        # create npz file name
        npz_file = sparam_file[-1].split(".")[0]

        # save as npz file
        np.savez(filedir / npz_file, f=sdata[0], s=sdata[1])

        # update xml file
        sparam_file.append(npz_file + ".npz")
        sparam_file = list(set(sparam_file))

        for each in node.iter("value"):
            if each.attrib["name"] == sparam_attr:
                each.text = ";".join(sparam_file)
        xml.write(filedir / lutfilename)

    if verbose:
        print("SParam data extracted in ", time.time() - start)
    return (sdata, npz_file)


def NetlistProcessor(spice_filepath, Network, libraries, c_, circuitData, verbose=True):
    """
    Processes a spice netlist to setup and simulate a circuit.

    Args:
        spice_filepath: Path to the spice netlist file.
        Network:

    """
    if verbose:
        for key, value in circuitData.items():
            print(key, str(value))

    # define frequency range and resolution
    freq = np.linspace(
        c_ / circuitData["sim_params"][0],
        c_ / circuitData["sim_params"][1],
        circuitData["sim_params"][2],
    )

    # create a circuit
    subckt = Network(network_id=circuitData["networkID"], f=freq)
    # get library
    all_libraries = dict(
        [
            each
            for each in inspect.getmembers(libraries, inspect.ismodule)
            if each[0][0] != "_"
        ]
    )
    libs_comps = {}
    for each_lib in list(set(circuitData["compLibs"])):
        # temp_comps = dict(inspect.getmembers(all_libraries[each_lib], inspect.isclass))
        if each_lib in all_libraries:
            # Using the libraries installed within the OPICS folder
            libs_comps[each_lib] = all_libraries[each_lib].component_factory
        else:
            opics_lib_name = "opics_"+each_lib
            try:
                # import the library named opics_xxx where xxx is the library name
                import importlib
                opics_lib = importlib.import_module(opics_lib_name)
                from importlib import reload
                opics_lib = reload(opics_lib)
                globals()[opics_lib_name] = opics_lib
                libs_comps[each_lib] = opics_lib.component_factory
            except:
                raise Exception ('Unable to import the library: %s' %opics_lib_name) 

    # add circuit components
    for i in range(len(circuitData["compModels"])):
        if len(circuitData["compModels"]) != len(circuitData["compLibs"]):
            raise Exception ('Unknown problem in OPICS and the compact model library. \nThe circuitData component model and component library lists have different lengths.')

        # get component model
        # check if the requested library is available
        if circuitData["compLibs"][i] in libs_comps:
            # check if the requested component is available in the library
            if circuitData["compModels"][i] in libs_comps[circuitData["compLibs"][i]]:
                comp_model = libs_comps[circuitData["compLibs"][i]][
                    circuitData["compModels"][i]
                    ]
            else:
                # Missing component
                raise Exception ('Missing OPICS compact model for component %s in library %s' % (circuitData["compModels"][i],circuitData["compLibs"][i]) ) 
        else:
            # Missing library
            raise Exception ('Missing OPICS compact model library for component %s in library %s' % (circuitData["compModels"][i],circuitData["compLibs"][i]) ) 
        # clean attributes
        cls_attrs = deepcopy(comp_model.cls_attrs)  # class attributes
        comp_attrs = circuitData["compAttrs"][i]  # component attributes
        # clean up attributes
        for each_attrs in cls_attrs.keys():
            if each_attrs in comp_attrs.keys():
                cls_attrs[each_attrs] = fromSI(comp_attrs[each_attrs])

        subckt.add_component(
            libs_comps[circuitData["compLibs"][i]][circuitData["compModels"][i]],
            params=cls_attrs,
            component_id=circuitData["compLabels"][i],
        )

    # add circuit netlist
    # subckt.global_netlist = circuitData["circuitNets"]
    subckt.global_netlist = {i:j for i,j in zip(circuitData['compLabels'], circuitData['circuitNets'])}

    # add unique net component connections
    subckt.current_connections = circuitData["circuitConns"]

    return subckt


class netlistParser:
    "A netlist parser to read spi files generated by SiEPIC tools"

    def __init__(self, mainfile_path: PosixPath) -> None:
        self.circuitComponents = []
        self.circuitConnections = []
        self.mainfile_path = mainfile_path

    def readfile(self) -> Dict[str, Any]:
        filepath = self.mainfile_path
        circuitID = ""
        inp = ""
        out = ""
        inp_net = 0
        out_net = []

        circuitLabels = []
        circuitModels = []
        circuitConns = []
        circuitNets = []
        componentLibs = []
        componentAttrs = []
        component_locations = []

        temp_file = open(filepath, "r")
        temp_lines = temp_file.readlines()

        free_node_idx = -1

        freq_data = []
        seek_component = 0
        seek_ona = 0
        orthogonal_ID = 0

        # extract circuit connectivity
        for each_line in temp_lines:
            each_line = re.sub(" +", " ", each_line.strip())  # remove empty lines
            if each_line.startswith("*"):
                continue
            else:
                each_line = "".join(
                    [
                        "".join(filter(None, each_section.split(" ")))
                        if ('"' in each_section)
                        else each_section
                        for each_section in re.split(
                            r"""("[^"]*"|'[^']*')""", each_line
                        )
                    ]
                )
                temp_data = each_line.split(" ")

                if len(temp_data) > 1:  # if line is not an empty one

                    MC_location = []

                    if temp_data[0] == ".subckt":
                        circuitID = temp_data[1]
                        inp = temp_data[2]
                        out = [temp_data[x] for x in range(3, len(temp_data))]
                        seek_component = 1

                    elif temp_data[0] == ".param":
                        continue

                    elif temp_data[0] == ".ends":
                        seek_component = 0

                    elif temp_data[0] == ".ona":
                        seek_ona = 1

                    elif seek_ona == 1:
                        # ONA related data
                        if len(temp_data) < 3:
                            temp_data = [0] + temp_data[-1].split("=")

                        if temp_data[1] == "orthogonal_identifier":
                            orthogonal_ID = int(temp_data[-1])

                        elif temp_data[1] == "start":
                            freq_data.append(float(temp_data[-1]))

                        elif temp_data[1] == "stop":
                            freq_data.append(float(temp_data[-1]))

                        elif temp_data[1] == "number_of_points":
                            freq_data.append(int(temp_data[-1]))

                    elif seek_component == 1:
                        # otherwise its component data
                        circuitLabels.append(temp_data[0])
                        temp_ports = []
                        found_ports = 0
                        found_library = 0
                        for i in range(1, len(temp_data)):
                            # if its an optical port
                            if (
                                "N$" in temp_data[i]
                                and "N$None".lower() != temp_data[i].lower()
                            ):
                                temp_ports.append(int(temp_data[i].replace("N$", "")))
                                found_ports = 1

                            elif "N$None".lower() == temp_data[i].lower():
                                temp_ports.append(free_node_idx)
                                free_node_idx -= 1
                                found_ports = 1

                            elif inp == temp_data[i]:
                                temp_ports.append(free_node_idx)
                                inp_net = free_node_idx
                                free_node_idx -= 1
                                found_ports = 1

                            elif out[0] == temp_data[i]:
                                temp_ports.append(free_node_idx)
                                out_net.append(free_node_idx)
                                free_node_idx -= 1

                                if len(out) > 1:
                                    out.pop(0)

                                if len(out) == 0:
                                    found_ports = 1

                            elif found_ports == 1 and "N$" not in temp_data[i]:
                                circuitModels.append(temp_data[i])
                                temp_cls_atrr = (
                                    {}
                                )  # deepcopy(lib[temp_data[i]].cls_attrs)
                                found_ports = -1

                            elif "lay" in temp_data[i] or "sch" in temp_data[i]:
                                if "lay" in temp_data[i]:
                                    MC_location.append(
                                        fromSI(temp_data[i].split("=")[-1]) * 1e6
                                    )

                                # ignore layout and schematic position data for now.
                                # adapt opics models to accept this data
                                # they are component parameters
                            elif "library" in temp_data[i]:
                                # cprint(temp_data[i])
                                temp_lib = (
                                    temp_data[i].replace('"', "").split("=")[1].split()
                                )
                                componentLibs.append(
                                    temp_lib[-1].split("/")[-1].lower()
                                )
                                found_library = 1

                            elif "=" in temp_data[i] and found_library == 1:
                                # if its a components' attribute
                                temp_attr = temp_data[i].split("=")
                                # print(temp_attr[0])
                                # if(temp_attr[0] in temp_cls_atrr):
                                temp_cls_atrr[temp_attr[0]] = temp_attr[1].strip('"')

                        componentAttrs.append(temp_cls_atrr)
                        circuitNets.append(temp_ports)
                        if bool(MC_location):
                            component_locations.append(MC_location)

        circuitConns = list(set(list(itertools.chain(*circuitNets))))
        # remove IOs from component connections' list
        circuitConns = [each for each in circuitConns if each >= 0]

        # return all data
        return {
            "circuitNets": circuitNets,
            "circuitConns": circuitConns,
            "compLibs": componentLibs,
            "compModels": circuitModels,
            "compLabels": circuitLabels,
            "compAttrs": componentAttrs,
            "compLocs": component_locations,
            "networkID": circuitID,
            "inp_net": inp_net,
            "out_net": out_net,
            "sim_params": freq_data,
            "OID": orthogonal_ID,
        }
