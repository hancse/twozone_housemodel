from dataclasses import dataclass, field


@dataclass
class CapacityNode:
    """component for a network node WITH (heat) capacity.

    Basic node in a topological network (graph),
    with (heat) capacity and a calculated temperature (or voltage)

    Attributes:
        label (str): node label e.g. "air", "wall".
        tag(int): numeric tag of node in total network system.
        cap (float): heat capacity or capacitance of node.
        temp (float): temperature or voltage of node.
    """
    label: str = field(default="label")
    tag: int = field(default=-1)
    cap: float = field(default=0.0)     # [J/K]
    temp: float = field(default=20.0)   # [K]


@dataclass
class FixedNode:
    """component for a fixed, external node.

    Basic external node, imposing a boundary condition to a topological network (graph),
    this basic component has no capacity (infinite capacity)
    and a predefined temperature (or voltage) that may change e.g. outdoor temperature.

    Attributes:
        label (str):   node label e.g. "outdoor", "indoor".
        connected_to (list): list of tags of CapacityNode objects, connected to this FixedNode.
        temp (float): predefined temperature or voltage of node.

    Note: if methods are defined this turns into a normal class object.
    """
    label: str
    connected_to: []
    temp: float        # [K]

    def update(self, new_value):
        self.temp = new_value


@dataclass
class CondEdge:
    """component for a network edge.

    Basic connecting edge in a topological network (graph),
    this basic component has a (thermal) conductance.

    Attributes:
        label (str):   node label e.g. "outdoor", "indoor".
        conn_nodes (list): list of tags of CapacityNode objects, connected by this CondEge.
        cond (float): conductance of node, in [W/K] or [A/V = S]

    Note: if methods are defined this turns into a normal class object.
    """
    label: str
    conn_nodes: []  # empty list (tuple, ndarray)
    cond: float
    # src: int
    # sink: int


""" notinuse
@dataclass
class PowerSource():
    label: str
    connected_to: []
    power: np.ndarray      # in [W]
"""


""" notinuse
@dataclass
class FlowEdge:
    label: str
    flow_rate: float  # [m^3/s]
    density: float    # [kg/m^3]
    cp: float         # J/(kg K)]
    heat_rate: float  # [J/s = W]
"""