# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from taccl.algorithm import Algorithm, Step
from taccl.topologies import TACCLTopology
from taccl.instance import Instance
from taccl.collectives import Collective, Chunk, Rank

import json
import warnings

def _sccl_object_hook(o):
    if not 'sccl_type' in o:
        return o
    if o['sccl_type'] == 'algorithm':
        input_map = { int(k): set(v) for k, v in o['input_map'].items() }
        output_map = { int(k): set(v) for k, v in o['output_map'].items() }
        return Algorithm(o['name'], o['collective'], o['topology'], o['instance'], o['steps'], input_map, output_map)
    if o['sccl_type'] == 'step':
        if len(o['sends'][0]) == 6:
            sends = [(addr, src, dst,t,l, redop) for addr, src, dst, t,l,redop in o['sends']]
        elif len(o['sends'][0]) == 5:
            sends = [(addr, src, dst,t,l) for addr, src, dst, t,l in o['sends']]
        elif len(o['sends'][0]) == 4:
            sends = [(addr, src, dst,t) for addr, src, dst, t in o['sends']]
        else:
            sends = [(addr, src, dst) for addr, src, dst in o['sends']]
        return Step(o['rounds'], sends)
    if o['sccl_type'] == 'collective':
        triggers = { (int(r), int(c)): v for r, rmap in o['triggers'].items() for c, v in rmap.items() }
        return Collective(o['name'], o['nodes'], o['chunks'], o['ranks'], triggers)
    if o['sccl_type'] == 'chunk':
        pre = set(o['pre'])
        post = set(o['post'])
        return Chunk(pre, post, o['addr'])
    if o['sccl_type'] == 'rank':
        pre = set(o['pre'])
        post = set(o['post'])
        return Rank(pre, post, o['id'])
    if o['sccl_type'] == 'topology':
        return TACCLTopology(o['name'], o['copies'], o['ngpus_per_node'], o['node_links'], o['node_invbws'], o['remote_invbw'], o['remote_alpha'], o['remote_beta'], o['internode_conn'], o['local_switches'])
    if o['sccl_type'] == 'instance':
        return Instance(o['steps'], o['extra_rounds'], o['chunks'], o['pipeline'], o['extra_memory'], o['allow_exchange'])
    warnings.warn('Unhandled sccl_type in JSON')

def SCCLDecoder():
    return json.JSONDecoder(object_hook=_sccl_object_hook)

class SCCLEncoder(json.JSONEncoder):
    def __init__(self):
        super().__init__()
    
    def default(self, o):
        if isinstance(o, Algorithm):
            input_map = { k: list(v) for k, v in o.input_map.items() }
            output_map = { k: list(v) for k, v in o.output_map.items() }
            return {
                'sccl_type': 'algorithm',
                'name': o.name,
                'instance': o.instance,
                'input_map': input_map,
                'output_map': output_map,
                'steps': o.steps,
                'collective': o.collective,
                'topology': o.topology,
            }
        if isinstance(o, Step):
            return {
                'sccl_type': 'step',
                'rounds': o.rounds,
                'sends': o.sends,
            }
        if isinstance(o, Collective):
            triggers = {}
            for (r, c), v in o._triggers.items():
                if not r in triggers:
                    triggers[r] = {}
                triggers[r][c] = v
            return {
                'sccl_type': 'collective',
                'name': o.name,
                'nodes': o.num_nodes,
                'chunks': o._chunks,
                'ranks': o._ranks,
                'triggers': triggers,
            }
        if isinstance(o, Chunk):
            return {
                'sccl_type': 'chunk',
                'pre': list(o.precondition),
                'post': list(o.postcondition),
                'addr': o.address,
            }
        if isinstance(o, Rank):
            return {
                'sccl_type': 'rank',
                'pre': list(o.precondition),
                'post': list(o.postcondition),
                'id': o.id,
            }
        if isinstance(o, TACCLTopology):
            return {
                'sccl_type': 'topology',
                'name': o.name,
                'copies': o.copies,
                'ngpus_per_node' : o.ngpus_per_node,
                'node_links' : o.node_links,
                'node_invbws' : o.node_invbws,
                'remote_invbw' : o.remote_invbw,
                'remote_alpha' : o.remote_alpha,
                'remote_beta' : o.remote_beta,
                'internode_conn' : o.internode_conn,
                'local_switches' : o.local_switches,
            }
        if isinstance(o, Instance):
            return {
                'sccl_type': 'instance',
                'steps': o.steps,
                'extra_rounds': o.extra_rounds,
                'chunks': o.chunks,
                'pipeline': o.pipeline,
                'extra_memory': o.extra_memory,
                'allow_exchange': o.allow_exchange,
            }
        return json.JSONEncoder.default(self, o)

def save_sccl_object(obj, filename):
    with open(filename, 'w') as f:
        f.write(SCCLEncoder().encode(obj))

def load_sccl_object(filename):
    with open(filename) as f:
        return SCCLDecoder().decode(f.read())
