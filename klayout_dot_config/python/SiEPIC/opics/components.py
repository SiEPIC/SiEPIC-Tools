import numpy as np
from scipy.interpolate import interp1d
from .utils import LUT_processor
from numpy import ndarray
from pathlib import PosixPath
from typing import Dict, List, Union
#from SiEPIC.opics.globals import F, C
from SiEPIC.opics.globals import C
import os
import binascii


class componentModel:
    """
    Defines the base component model class used to create new components\
            for a library.

    Args:
        f (numpy.ndarray): Frequency datapoints.
        nports (int): Number of ports in the component. Defaults to 2.
        s (numpy.ndarray, optional): S-parameter data. Defaults to None.
        data_folder (pathlib.Path): The location of the data folder\
            containing s-parameter data files and the XML look-up-table.
        filename (str): Name of the XML look-up-table file.
        sparam_attr (str, optional): Look-up-table attribute\
                Defaults to None.
    """

    def __init__(
        self,
        f: ndarray = None,
        nports: int = 2,
        s: ndarray = None,
        data_folder: PosixPath = None,
        filename: str = None,
        sparam_attr: str = None,
        **kwargs
    ) -> None:

        self.f = f
        if self.f is None:
#            raise Exception ('Frequency range not defined, in opics/network')
            print ('Frequency range not defined, in opics/network')
#            self.f = F

        self.C = C

        self.s = s
        if s is None:
            self.s = np.array((nports, nports))

        #self.lambda_ = self.C * 1e6 / self.f
        self.componentParameters = []
        self.component_id = str(binascii.hexlify(os.urandom(4)))[2:-1]
        self.nports = nports

        self.port_references = {}
        for _ in range(self.nports):
            self.port_references[_] = _

        self.sparam_attr = sparam_attr
        self.sparam_file = filename

        for key, value in kwargs.items():
            self.componentParameters.append([key, str(value)])

    def load_sparameters(self, data_folder: PosixPath, filename: str) -> ndarray:
        """
        Loads sparameters either from an npz file or from a raw sparam\
             file using a look-up table.

        Args:
            data_folder: Directory path of the data folder containing\
                 s-parameter data files and a look up table.
            filename: Name of the XML look-up-table file.

        Returns:
            sparameters: Array of the component's s-parameters.
        """

        if ".npz" in filename:
            componentData = np.load(data_folder / filename)
            return self.interpolate_sparameters(
                self.f, componentData["f"], componentData["f"]
            )
        else:
            componentData, self.sparam_file = LUT_processor(
                data_folder,
                filename,
                self.componentParameters,
                self.nports,
                self.sparam_attr,
            )
            return self.interpolate_sparameters(
                self.f, componentData[0], componentData[1]
            )

    def set_port_reference(self, port_number: int, port_name: str) -> None:
        """
        Allows for custom naming of component ports

        Args:
            port_number: Port number.
            port_name: Custom port name.
        """
        self.port_references[port_name] = port_number
        self.port_references[port_number] = port_name

    def interpolate_sparameters(
        self, target_f: ndarray, source_f: ndarray, source_s: ndarray
    ) -> ndarray:
        """
        Cubic interpolation of the component sparameter data to match the\
             desired simulation frequency range.

        Args:
            target_f: The target frequency range onto which\
                 the s-parameters will be interpereted on.
            source_f: The source frequency range that the\
                 component data has stored.
            source_s: The source s-parameters that the\
                 component data has stored.

        Returns:
            sparameters: Interpolated s-parameters value\
                 over the target frequency range.
        """

        fill1 = source_s[0]
        fill2 = source_s[-1]
#        func = interp1d(source_f, source_s, kind="cubic", axis=0, bounds_error=False, fill_value = 0) # results in a truncated spectrum
#        func = interp1d(source_f, source_s, kind="cubic", axis=0, bounds_error=False, fill_value = (1e-3,1e-2))
        func = interp1d(source_f, source_s, kind="cubic", axis=0, bounds_error=False, fill_value = (fill2,fill1))
#        func = interp1d(source_f, source_s, kind="cubic", axis=0, fill_value = "extrapolate")
#        func = interp1d(source_f, source_s, kind="cubic", axis=0, fill_value = 0)
        return func(target_f)

    def write_sparameters(
        self, dirpath: PosixPath, filename: str, f_data: ndarray, s_data: ndarray
    ) -> None:
        """Export the simulated s-parameters to a file.

        Args:
            dirpath: Directory path.
            filename: Name of the file.
            f_data: Frequency data.
            s_data: S-parameter data.
        """
        with open(dirpath / filename, "w") as datafile_id:
            datalen = s_data.shape[0]

            for i in range(s_data.shape[1]):
                for j in range(s_data.shape[2]):
                    datafile_id.write(
                        "('port %d','TE',1,'port %d',1,'transmission')\n" % (i, j)
                    )
                    datafile_id.write("(%d,3)\n" % (datalen))

                    temp_data = s_data[:, i, j]
                    data = np.array([f_data, np.abs(temp_data), np.angle(temp_data)])
                    data = data.T
                    np.savetxt(datafile_id, data, fmt=["%d", "%f", "%f"])

    def get_data(
        self, ports: List[List[int]] = None, xscale: str = "freq", yscale: str = "log"
    ) -> Dict[str, Union[ndarray, str]]:
        """Get the S-parameters data for specific [input,output] port\
             combinations, to be used for plotting functionalities.
        (WARNING: unused, to be used in plot_sparameters)

        Args:
            ports: List of lists that contains the desired\
                 S-parameters, e.g., [[1,1],[1,2],[2,1],[2,2]].\
                     Defaults to None.
            xscale: Plotting x axis label. Defaults to "freq".
            yscale: Plotting Y axis label. Defaults to "log".

        Returns:
            temp_data: Dictionary containing the plotting information\
                 to be used, including S-parameters data and plotting labels.
        """
        temp_data = {}  # reformat data in an array

        ports_ = []  # ports the plot

        if xscale == "freq":
            x_data = self.f
            xlabel = "Frequency (Hz)"
        else:
            x_data = self.C * 1e6 / self.f
            xlabel = "Wavelength (um)"

        temp_data["xdata"] = x_data
        temp_data["xunit"] = xlabel

        if ports is None:
            nports = self.s.shape[-1]
            for i in range(nports):
                for j in range(nports):
                    ports_.append("S_%d_%d" % (i, j))
        else:
            ports_ = ["S_%d_%d" % (each[0], each[1]) for each in ports]

        for each_port in ports_:
            _, i, j = each_port.split("_")
            if yscale == "log":
                temp_data[each_port] = 10 * np.log10(
                    np.square(np.abs(self.s[:, int(i), int(j)]))
                )
                temp_data["yunit"] = "dB"
            elif yscale == "abs":
                temp_data[each_port] = np.abs(self.s[:, int(i), int(j)])
                temp_data["yunit"] = "abs"
            elif yscale == "abs_sq":
                temp_data[each_port] = np.square(np.abs(self.s[:, int(i), int(j)]))
                temp_data["yunit"] = "abs_sq"

        return temp_data

    def plot_sparameters(
        self,
        ports: List[List[int]] = None,
        show_freq: bool = True,
        scale: str = "log",
        interactive: bool = False,
    ):
        """Plot the component's S-parameters.

        Args:
            ports: List of lists that contains the desired\
                 S-parameters, e.g., [[1,1],[1,2],[2,1],[2,2]].\
                      Defaults to None.
            show_freq: Flag to determine whether to plot\
                 with respect to frequency or wavelength. Defaults to True.
            scale: Plotting y axis scale, options available:\
                 ["log", "abs", "abs_sq"]. Defaults to "log".
            interactive: Make the plots interactive or not.
        """

        ports_ = []  # ports the plot

        if show_freq:
            x_data = self.f
            xlabel = "Frequency (Hz)"
        else:
            x_data = self.C * 1e6 / self.f
            xlabel = "Wavelength (um)"

        if ports is None:
            nports = self.s.shape[-1]
            for i in range(nports):
                for j in range(nports):
                    ports_.append("S_%d_%d" % (i, j))
        else:
            ports_ = ["S_%d_%d" % (each[0], each[1]) for each in ports]

        if not interactive:
            import matplotlib.pyplot as plt
            for each_port in ports_:
                _, i, j = each_port.split("_")
                if scale == "log":
                    plt.plot(
                        x_data,
                        10 * np.log10(np.square(np.abs(self.s[:, int(i), int(j)]))),
                    )
                    plt.ylabel("Transmission (dB)")
                elif scale == "abs":
                    plt.plot(x_data, np.abs(self.s[:, int(i), int(j)]))
                    plt.ylabel("Transmission (normalized)")
                elif scale == "abs_sq":
                    plt.plot(x_data, np.square(np.abs(self.s[:, int(i), int(j)])))
                    plt.ylabel("Transmission (normalized^2)")
            plt.xlabel(xlabel)
            plt.xlim(left=np.min(x_data), right=np.max(x_data))
            plt.tight_layout()
            plt.legend(ports_)
            plt.show()
        else:
            import holoviews as hv
            import pandas as pd
            from bokeh.plotting import show

            hv.extension("bokeh")
            temp_data = self.get_data(ports=ports, xscale="lambda", yscale=scale)
            filtered_s = dict(
                [[key, temp_data[key]] for key in temp_data.keys() if "unit" not in key]
            )
            df = pd.DataFrame.from_dict(filtered_s)
            master_plot = None
            for each_ydata in ports_:
                if master_plot is None:
                    master_plot = hv.Curve(
                        df, "xdata", each_ydata, label=each_ydata
                    ).opts(tools=["hover"])
                else:
                    curve = hv.Curve(df, "xdata", each_ydata, label=each_ydata).opts(
                        tools=["hover"]
                    )
                    master_plot = master_plot * curve

            master_plot.opts(
                ylabel=temp_data["yunit"],
                xlabel=temp_data["xunit"],
                responsive=True,
                min_height=400,
                min_width=600,
                fontscale=1.5,
                max_width=800,
                max_height=600,
            )

            show(hv.render(master_plot))
