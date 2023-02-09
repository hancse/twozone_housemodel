from dataclasses import dataclass, field


@dataclass
class CapacityNode:
    label: str = field(default="label")
    tag: int = field(default=-1)
    cap: float = field(default=0.0)     # [J/K]
    temp: float = field(default=20.0)   # [K]


@dataclass
class FixedNode:   # external node?
    label: str
    connected_to: []
    temp: float        # [K]
# if methods are defined this turns into a normal class object

    def update(self, new_value):
        self.temp = new_value


@dataclass
class CondEdge:
    label: str
    conn_nodes: []  # empty list (tuple, ndarray)
    cond: float     # [W/K]
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