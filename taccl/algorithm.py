# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Step(object):
    rounds: int
    sends: list

class Algorithm(object):
    def __init__(self, name, collective, topology, instance, steps, input_map = {}, output_map = {}, cont=False):
        self.name = name
        self.topology = topology
        self.collective = collective
        self.instance = instance
        self.steps = steps
        self.input_map = input_map
        self.output_map = output_map
        self.cont = cont

        self._update_link_utilizations()
        if not cont:
            self._check_bandwidth_constraints()
        else:
            self._check_real_bandwidth_constraints()

    @classmethod
    def make_implementation(cls, collective, topology, instance, steps, cont=False, suffix=""):
        # Figure out input and output addresses
        input_map = {}
        output_map = {}
        for rank in collective.ranks():
            input_addrs = set()
            output_addrs = set()
            for chunk in collective.chunks():
                # An address is an input address if any of its chunks is in the precondition
                if collective.precondition(rank, chunk):
                    input_addrs.add(collective.address(chunk))
                # An address is an output address if any of its chunks is in the postcondition
                if collective.postcondition(rank, chunk):
                    output_addrs.add(collective.address(chunk))
            if len(input_addrs) > 0:
                input_map[rank] = input_addrs
            if len(output_addrs) > 0:
                output_map[rank] = output_addrs

        # Concatenate collective and topology names plus instance arguments to create a name
        name = f'{collective.name}-{topology.name}-{instance}{suffix}'

        algo = cls(name, collective, topology, instance, steps, input_map, output_map, cont)
        algo.check_implements(collective)
        if instance.extra_rounds > 0:
            used_extra_rounds = algo.extra_rounds()
            if used_extra_rounds > instance.extra_rounds:
                raise ValueError(f'steps use {used_extra_rounds} extra rounds but only {instance.extra_rounds} were allowed')
        return algo

    def ranks(self):
        return range(self.topology.num_nodes())
    
    def num_steps(self):
        return len(self.steps)

    def extra_rounds(self):
        rounds = 0
        for step in self.steps:
            rounds += step.rounds
        return rounds - self.num_steps()

    def is_pipelined(self):
        return self.instance.pipeline != None

    def check_implements(self, collective):
        if self.topology.num_nodes() != collective.num_nodes:
            raise RuntimeError('topology and collective have different number of nodes')
        # Find which chunks will be sent from an address
        chunks_at_address = defaultdict(list)
        for chunk in collective.chunks():
            chunks_at_address[collective.address(chunk)].append(chunk)
        # State records if a rank holds a chunk
        def idx(rank, chunk):
            return rank * collective.num_chunks + chunk
        state = [False] * (collective.num_nodes * collective.num_chunks)
        # Initialize state from precondition
        for rank in collective.ranks():
            for chunk in collective.chunks():
                state[idx(rank, chunk)] = collective.precondition(rank, chunk)
        # Propagate state through sends of every step
        for step in self.steps:
            next_state = state.copy()
            if len(step.sends[0]) == 5:
                for addr, src, dst, _, _ in step.sends:
                    for chunk in chunks_at_address[addr]:
                        next_state[idx(dst, chunk)] |= state[idx(src, chunk)]
            elif len(step.sends[0]) == 6:
                for addr, src, dst, _, _, _ in step.sends:
                    for chunk in chunks_at_address[addr]:
                        next_state[idx(dst, chunk)] |= state[idx(src, chunk)]
            else:
                for addr, src, dst in step.sends:
                    for chunk in chunks_at_address[addr]:
                        next_state[idx(dst, chunk)] |= state[idx(src, chunk)]
            state = next_state
        # Check that the postcondition holds
        for rank in collective.ranks():
            for chunk in collective.chunks():
                # print(rank, chunk, state[idx(rank, chunk)])
                if collective.postcondition(rank, chunk) and not state[idx(rank, chunk)]:
                    raise RuntimeError(f'rank {rank} does not get chunk {chunk} as required by the postcondition')

    def _update_link_utilizations(self):
        self._link_utilizations = []
        ranks = range(self.topology.num_nodes())
        for step in self.steps:
            step_utilizations = [[0 for _ in ranks] for _ in ranks]
            if len(step.sends[0]) == 5:
                for addr, src, dst, _, _ in step.sends:
                    step_utilizations[dst][src] += 1 # Same order as topology
            elif len(step.sends[0]) == 6:
                for addr, src, dst, _, _, _ in step.sends:
                    step_utilizations[dst][src] += 1 # Same order as topology
            else:
                for addr, src, dst in step.sends:
                    step_utilizations[dst][src] += 1 # Same order as topology
            self._link_utilizations.append(step_utilizations)

    def _check_bandwidth_constraints(self):
        for srcs, dsts, bw, name in self.topology.bandwidth_constraints():
            for step_num, step in enumerate(self.steps):
                util = 0
                for dst in dsts:
                    for src in srcs:
                        if self.is_pipelined():
                            for overlapping_step in range(step_num, len(self.steps), self.instance.pipeline):
                                util += self._link_utilizations[overlapping_step][dst][src]
                        else:
                            util += self._link_utilizations[step_num][dst][src]
                assert util <= bw * step.rounds, \
                    f'Step {step_num} uses {util} bandwidth but constraint {name} only allows for {bw * step.rounds} bandwidth (when rounds={step.rounds}).'

    def _check_real_bandwidth_constraints(self):
        for srcs, dsts, bw, l, name in self.topology.real_bandwidth_constraints():
            for step_num, step in enumerate(self.steps):
                util = 0
                for dst in dsts:
                    for src in srcs:
                        if self.is_pipelined():
                            for overlapping_step in range(step_num, len(self.steps), self.instance.pipeline):
                                util += self._link_utilizations[overlapping_step][dst][src]
                        else:
                            util += self._link_utilizations[step_num][dst][src]
                assert util * bw <= step.rounds, \
                    f'Step {step_num} uses {util * bw} time but constraint {name} only allows for {step.rounds} time (when rounds={step.rounds}).'


    def __str__(self):
        s = ''
        for i, step in enumerate(self.steps):
            if i != 0:
                s += '\n'
            if step.rounds > 1:
                s += f'(step {i+1}, rounds={step.rounds}) '
            else:
                s += f'(step {i+1}) '
            if len(step.sends[0]) == 5:
                s += ', '.join([f'{chunk}:{src}→{dst}' for chunk, src, dst, _, _ in step.sends])
            elif len(step.sends[0]) == 6:
                s += ', '.join([f'{chunk}:{src}→{dst}' for chunk, src, dst, _, _, _ in step.sends])
            else:
                s += ', '.join([f'{chunk}:{src}→{dst}' for chunk, src, dst in step.sends])
        return s
