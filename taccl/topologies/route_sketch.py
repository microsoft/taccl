from dataclasses import dataclass

@dataclass
class IntraNode:
    strategy: str

@dataclass
class IntraNode_Switch(IntraNode):
    switches: list
    switch_hyperedge_strategy: list

@dataclass
class IntraNode_RelaySwitch(IntraNode):
    relayed_switch_conn: dict

@dataclass
class InterNode:
    pass

@dataclass
class InterNode_Switch(InterNode):
    switches: list
    switch_hyperedge_strategy: list

@dataclass
class InterNode_Relay(InterNode):
    internode_conn: dict
    gpus_to_sender_rev_map: dict
    enforce_ordering: bool

@dataclass
class MultiNode:
    strategy: list
    nnodes: list
    group_size: list

@dataclass
class Symmetry:
    offsets: list

@dataclass
class HyperParameter:
    chunkup: int
    heuristic: int

@dataclass
class RouteSketch:
    intranode: IntraNode
    internode: InterNode
    multinode: MultiNode
    symmetry: Symmetry
    hyperparameters: HyperParameter