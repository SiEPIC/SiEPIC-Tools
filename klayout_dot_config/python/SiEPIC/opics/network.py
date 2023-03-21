import os
import binascii
from typing import List, Optional, Dict, Union
from numpy import ndarray
from SiEPIC.opics.sparam_ops import connect_s
from SiEPIC.opics.components import componentModel
#from SiEPIC.opics.globals import F
import multiprocessing as mp


def solve_tasks(
    data: List,
):
    """
    Simulates a connection, either shared by two different components or by the same component.

    Args:
        data:   A list with the following elements:\
                [components_connected, \
                net_to_port_data, \
                components_nets].
    """
    components, ntp, nets = data
    # If pin occurances are in the same component:
    if ntp[0] == ntp[2]:
        # print(t_components[ca].s.shape)
        new_s = connect_s(
            components[0].s,
            ntp[1],
            None,
            ntp[3],
            create_composite_matrix=False,
        )
        components[0].s = new_s
        new_component = components[0]

        new_net = nets[0]
        p1, p2 = new_net[ntp[1]], new_net[ntp[3]]

        # delete both port references
        new_net.remove(p1)
        new_net.remove(p2)

        new_component.nports = len(new_net)
        port_references = {}
        for _ in range(new_component.nports):
            port_references[_] = _
        new_component.port_references = port_references

        return new_component, new_net

    # If pin occurances are in different components:
    else:
#        combination_f = F
        combination_f = None
        combination_s = connect_s(components[0].s, ntp[1], components[1].s, ntp[3])

        # nets of the new component
        net1, net2 = nets[0], nets[1]
        del net1[ntp[1]], net2[ntp[3]]
        new_net = net1 + net2

        # create new component
        new_component = componentModel(
            f=combination_f, s=combination_s, nets=new_net, nports=len(new_net)
        )
        return new_component, new_net


class Network:
    """
    Defines a circuit or a network.

    Args:
        network_id: Define the network name.
        f: Frequency data points.
        mp_config: Enable/Disable multiprocessing (disabled by default);\
                    Expects the following information:\n
                        1. "enabled" : bool - enable/disable multiprocessing,\n
                        2. "proc_count": int - process count\n
                        3. "close_pool": bool - Should the solver terminate all the processes after the simulation is finished.
    """

    def __init__(
        self,
        network_id: Optional[str] = None,
        f: Optional[ndarray] = None,
        mp_config: Dict = {"enabled": False, "proc_count": 0, "close_pool": False},
    ) -> None:

        self.f = f
        if self.f is None:
            print('Frequency range not defined, in opics/network')
#            raise Exception ('Frequency range not defined, in opics/network')
#            self.f = F

        self.network_id = (
            network_id if network_id else str(binascii.hexlify(os.urandom(4)))[2:-1]
        )
        self.current_components = {}
        self.current_connections = []
        self.global_netlist = {}
        self.port_references = {}
        self.sim_result = None

        self.mp_config = mp_config

        if self.mp_config["enabled"]:
            if "close_pool" not in self.mp_config:
                self.mp_config["close_pool"] = True

            if (
                "proc_count" not in self.mp_config
                or type(self.mp_config["proc_count"]) != int
            ):
                self.mp_config["proc_count"] = 0

            # create process pool
            if self.mp_config["proc_count"] == 0:
                self.pool = mp.Pool()
            else:
                self.pool = mp.Pool(processes=self.mp_config["proc_count"])
            print("OPICS multiprocessing is enabled.")

    def add_component(
        self,
        component: componentModel,
        params: Dict = {},
        component_id: Optional[str] = None,
    ):
        """
        Adds component to a network.

        Args:
            component: An instance of componentModel class.
            params: Component parameter values.
            component_id: Custom component id tag.
        """

        if isinstance(component, componentModel):
            self.current_components[component.component_id] = component
            return component

        if "f" not in params:
            params["f"] = self.f

        temp_component = component(**params)

        temp_component.component_id = (
            temp_component.component_id
            + "_"
            + str(binascii.hexlify(os.urandom(4)))[2:-1]
            if component_id is None
            else component_id
        )

        self.current_components[temp_component.component_id] = temp_component

        return temp_component

    def connect(
        self,
        component_A_id: Union[str, componentModel],
        port_A: int,
        component_B_id: Union[str, componentModel],
        port_B: int,
    ):
        """
        Connects two components together.

        Args:
            component_A_id: A component ID or an instance of componentModel class.
            port_A: Port number of component_A
            component_B_id: A component ID or an instance of componentModel class.
            port_B: Port number of component_B
        """
        if component_A_id.__class__.__bases__[-1] == componentModel:
            component_A_id = component_A_id.component_id
        if component_B_id.__class__.__bases__[-1] == componentModel:
            component_B_id = component_B_id.component_id

        if type(port_A) == str:
            port_A = self.current_components[component_A_id].port_references[port_A]
        if type(port_B) == str:
            port_B = self.current_components[component_B_id].port_references[port_B]

        self.current_connections.append(
            [
                component_A_id,
                port_A,
                component_B_id,
                port_B,
            ]
        )

    def initiate_global_netlist(self):
        """
        Initiates a global netlist with negative indices, \
            overwrite indices that are used in the circuit\
                 with positive values.
        """

        gnetlist = {}
        net_start = 0
        # for each component
        component_ids = self.current_components.keys()
        for component_id in component_ids:
            temp_net = []
            # for each port of component
            for each_port in range(self.current_components[component_id].s.shape[-1]):
                net_start -= 1
                temp_net.append(net_start)
                # if the port has a custom name
                if (
                    self.current_components[component_id].port_references[each_port]
                    != each_port
                ):
                    self.port_references[net_start] = self.current_components[
                        component_id
                    ].port_references[each_port]

            gnetlist[component_id] = temp_net

        for i in range(len(self.current_connections)):
            each_conn = self.current_connections[i]
            # update connected ports to positive values, marked with net_ids
            gnetlist[each_conn[0]][each_conn[1]] = i
            gnetlist[each_conn[2]][each_conn[3]] = i

        self.global_netlist = gnetlist

    def global_to_local_ports(self, net_id: int, nets: List[List[int]]) -> List[int]:
        """
        Maps the net_id to components and their local port numbers.

        Args:
            net_id: Net id reference.
            nets: Nets
        """
        _component_ids = self.current_components.keys()

        # Get components associated with the net_id
        filtered_components = [
            each_comp_id
            for each_comp_id in _component_ids
            if net_id in nets[each_comp_id]
        ]

        if len(filtered_components) == 1:
            filtered_components += filtered_components

        net_idx = []
        for each_comp in filtered_components:
            net_idx += [i for i, x in enumerate(nets[each_comp]) if x == net_id]

        return [filtered_components[0], net_idx[0], filtered_components[1], net_idx[1]]

    def simulate_network(self) -> componentModel:
        """
        Triggers the simulation
        """

        # create global netlist
        if not bool(self.global_netlist):
            self.initiate_global_netlist()

        # check if all the components are connected
        _not_connected = set()
        _component_names = self.global_netlist.keys()

        for _ in _component_names:
            _temp_component_pins = self.global_netlist[_]
            _temp_component_pins = [each_pin >= 0 for each_pin in _temp_component_pins]
            if True not in _temp_component_pins:
                _not_connected.add(_)

        if bool(_not_connected):
            self.global_netlist = {}
            raise RuntimeError("Some components are not connected.")

        t_components = self.current_components
        t_nets = self.global_netlist
        
        # Lukas: this line worked for a single circuit, but failed
        # when there were two separate circuits on the same layout
        # Lukas: I don't know why these don't have the same type
        if type(self.current_connections[0])==int:
            # this is the case for SPICE imported files, as fixed by Mustafa
            # ex. [10, 11, 12, 13]
            t_connections = self.current_connections
        else:
            # this is the case for circuits created using Python, circuit.connect, etc, by Jaspreet
            # [['Ebeam_GC_48a2e40d', 1, 'Ebeam_Y_25eaf721', 0], ['Ebeam_Y_25eaf721', 1, 'Ebeam_WG_0fc722dc', 0], 
            t_connections = [i for i in range(len(self.current_connections))]
            

        _connections_in_use = set()

        while len(t_connections) > 0:

            # track components and connections in use
            _components_in_use = set()
            _nets_in_use = set()
            _task_bundle = []

            # ------------ Step 1: Create Task Bundles------------------
            # for loop to iterate over connections
            for _connection in t_connections:
                if _connection not in _connections_in_use:
                    # get components and port indexes
                    net_to_port = self.global_to_local_ports(_connection, t_nets)

                    # components are already being used in another net, skip this connection
                    if (
                        net_to_port[0] in _components_in_use
                        or net_to_port[2] in _components_in_use
                    ):
                        continue

                    # lock components, nets, and connections to prevent from being used in other threads
                    _connections_in_use.add(_connection)
                    _components_in_use.add(net_to_port[0])
                    _components_in_use.add(net_to_port[2])
                    _nets_in_use.add(tuple(t_nets[net_to_port[0]]))
                    _nets_in_use.add(tuple(t_nets[net_to_port[2]]))

                    # -------- Step 2: add components to tasks bundles -----------
                    if net_to_port[0] == net_to_port[2]:
                        # if the both components are the same
                        _task_bundle.append(
                            [
                                [t_components[net_to_port[0]], None],
                                net_to_port,
                                [t_nets[net_to_port[0]], t_nets[net_to_port[2]]],
                            ]
                        )

                    else:
                        _task_bundle.append(
                            [
                                [
                                    t_components[net_to_port[0]],
                                    t_components[net_to_port[2]],
                                ],
                                net_to_port,
                                [t_nets[net_to_port[0]], t_nets[net_to_port[2]]],
                            ]
                        )

            # ------- Step 3: Remove components, nets, connection ids ----------
            for _ in _components_in_use:
                if _ in t_components:
                    t_components.pop(_)

            for _ in list(t_nets.keys()):
                if tuple(t_nets[_]) in _nets_in_use:
                    t_nets.pop(_)

            t_connections = [
                each_conn
                for each_conn in t_connections
                if each_conn not in _connections_in_use
            ]

            # ------- solve tasks and merge results -----------
            if self.mp_config["enabled"]:
                results = self.pool.map(solve_tasks, _task_bundle)
            else:
                results = [solve_tasks(_) for _ in _task_bundle]

            # merge results
            for each_result in results:
                t_components[each_result[0].component_id] = each_result[0]
                t_nets[each_result[0].component_id] = each_result[1]

        if self.mp_config["enabled"]:
            if self.mp_config["close_pool"]:
                self.pool.close()
                self.pool.join()

        t_components[list(t_components.keys())[-1]].component_id = self.network_id
        self.sim_result = t_components[list(t_components.keys())[-1]]
        self.current_connections = []
        return t_components[list(t_components.keys())[-1]]

    def enable_mp(self, process_count: int = 0, close_pool: bool = True):
        """
        Enables OPICS multiprocessing

        Args:
            process_count: Number of processes to start. Leave the default value if not sure (let the system decide). Otherwise, use `multiprocessing.cpu_count()` to know the maximum number of processes that can be run safely.
            close_pool: Whether to terminate all the processes after the simulation is done.
        """
        if not self.mp_config["enabled"]:
            self.mp_config["enabled"] = True
            self.mp_config["proc_count"] = process_count
            self.mp_config["close_pool"] = close_pool
            if self.mp_config["proc_count"] == 0:
                self.pool = mp.Pool()
            else:
                self.pool = mp.Pool(processes=self.mp_config["proc_count"])
            print("OPICS multiprocessing is enabled.")

    def disable_mp(self):
        """
        Disables OPICS multiprocessing
        """
        if self.mp_config["enabled"]:
            # close all processes.
            self.pool.close()
            self.pool.join()
            self.mp_config["enabled"] = False
            print("OPICS multiprocessing is disabled.")


# mp helper functions
def bulk_add_component(network: Network, components_data: List[Dict]):
    """
    Allows for bulk adding of components

    Args:
        network: Network to add components to.
        components_data: A list of dictionaries including component class reference, parameter data, and component id
    """
    if network.mp_config["enabled"]:
        temp_comps = network.pool.map(inst_components, components_data)
    else:
        temp_comps = [
            inst_components(each_component) for each_component in components_data
        ]

    # add temporary component instances to the network
    for each_component in temp_comps:
        network.current_components[each_component.component_id] = each_component


def inst_components(component_data: dict):
    """
    Given a component class, component parameter data, and component id, returns an instance of the component class.

    Args:
        component_data: {"component": component_class, "params": component_parameters, "component_id", custom_component_id}\n
                        Example: {"component": ebeam.Waveguide, "params":{"f": circuit.f, "length": 10e-6}, "component_id": "test_waveguide"}

    """

    temp_component = component_data["component"](**component_data["params"])

    temp_component.component_id = (
        temp_component.component_id + "_" + str(binascii.hexlify(os.urandom(4)))[2:-1]
        if "component_id" not in component_data
        else component_data["component_id"]
    )

    return temp_component
