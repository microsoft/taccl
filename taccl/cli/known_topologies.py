# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import taccl.topologies as topologies
from pathlib import Path
import sys

class KnownTopologies:
    def __init__(self, parser, tag=''):
        self.parser = parser
        self.tag = tag
        self.constructors = {
            'HubAndSpoke': self._topo(topologies.hub_and_spoke),
            'DGX2': self._topo(topologies.dgx2),
            'NDv2': self._topo(topologies.ndv2),
            'custom': self._topo(topologies.custom),
        }
        self.parser.add_argument(f'topology{tag}', type=str, choices=self.constructors.keys(), help=f'topology {tag}')
        self.parser.add_argument(f'--topology-file{tag}', type=str, default=None, help=f'profiled topology')

    def _topology(self, args):
        return vars(args)[f'topology{self.tag}']

    def _topology_file(self, args):
        input_str = vars(args)[f'topology_file{self.tag}']
        if input_str is None:
            self.parser.error(f'--topology-file{self.tag} is required')
            exit(1)

        input_file = Path(input_str)
        if not input_file.exists():
            print(f'error: input file not found: {input_file}', file=sys.stderr)
            exit(1)

        return input_file

    def create(self, args):
        topology = self.constructors[self._topology(args)](args)
        return topology

    def _topo(self, Cls):
        def make(args):
            return Cls(self._topology_file(args))
        return make

