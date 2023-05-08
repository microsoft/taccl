# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from collections import defaultdict
import math

class NodeTopology(object):
    def __init__(self, name, links, alpha, betas, invbws, nics_per_node, remote_invbw, remote_alpha, remote_beta):
        self.name = name
        self.links = links
        self.alpha = alpha
        self.betas = betas
        self.invbws= invbws
        self.nics_per_node = nics_per_node
        self.remote_invbw = remote_invbw
        self.remote_alpha = remote_alpha
        self.remote_beta = remote_beta

def _add_ext_edge(links, ngpus_per_node, ngpus, copies, internode_conn, remote_link, link_split=None, multinode_split=1):
    new_links = [
        [
            links[dst % ngpus_per_node][src % ngpus_per_node] 
            if src // ngpus_per_node == dst // ngpus_per_node
            else 0
            for src in range(ngpus)
        ] for dst in range(ngpus)
    ]

    for i in range(copies):
        for j in range(copies):
            if i == j:
                continue
            for s in internode_conn:
                for r in internode_conn[s]:
                    src = int(s) + i * ngpus_per_node
                    dst = r + j * ngpus_per_node
                    if link_split is not None:
                        rlink = remote_link * link_split[s] * multinode_split
                    else:

                        rlink = remote_link * multinode_split
                    new_links[dst][src] = rlink
    return new_links

def _add_ext_switches(invbws, ngpus_per_node):
    swt = []
    for r in range(len(invbws)):
        c = r // ngpus_per_node
        node = r % ngpus_per_node
        dsts = [dst for dst in range(len(invbws)) if ((r//ngpus_per_node != dst//ngpus_per_node) and (invbws[dst][r] > 0))]
        srcs = [src for src in range(len(invbws)) if ((r//ngpus_per_node != src//ngpus_per_node) and (invbws[r][src] > 0))]
        invbw = None
        if len(dsts) != 0:
            invbw = invbws[dsts[0]][r]
        if len(srcs) != 0:
            invbw = invbws[r][srcs[0]]
        if invbw is not None:
            swt.append(([r], dsts, 1, invbw, f'copy_{c}_node_{node}_out_remot'))
            swt.append((srcs, [r], 1, invbw, f'copy_{c}_node_{node}_in_remot'))
    return swt

def _make_switch(switches, node_beta, copies, ngpus_per_node):
    new_switches = []
    num_switches = len(switches)
    for i in range(num_switches):
        swt = []
        for c in range(copies):
            for node in switches[i]:
                dist_node = node + c * ngpus_per_node
                dist_others = [other + c * ngpus_per_node for other in switches[i] if other != node]
                invbw = node_beta[dist_others[0]][dist_node]
                for o in dist_others:
                    assert node_beta[o][dist_node] == invbw
                swt.append(([dist_node], dist_others, 1, invbw, f'copy_{c}_node_{node}_swt_{i}_out_local'))
                swt.append((dist_others, [dist_node], 1, invbw, f'copy_{c}_node_{node}_swt_{i}_in_local'))
        new_switches.append(swt)
    return new_switches

# TODO: whether to make relays switch and have same b/w or have them share b/w
# will be decided by multinode-sketch and we need to handle the way links b/w is assigned
# Right now, we assume that there will only be a single group of nodes, allowing the link to
# be = group_size * link
class TACCLTopology(object):
    def __init__(self, name, copies, ngpus_per_node, node_links,
            node_invbws, remote_invbw, remote_alpha, remote_beta,
            internode_conn, switches=[]):
        self.name = name
        self.copies = copies
        self.ngpus_per_node = ngpus_per_node
        self.ngpus = copies * ngpus_per_node
        self.node_links = node_links
        self.node_invbws = node_invbws
        self.invbws = node_invbws
        self.internode_conn = internode_conn
        self.base_gpus = []
        self.local_switches = switches
        self.remote_beta = remote_beta
        ext_switches = []
        base_gpu = 0
        for c in range(copies):
            self.base_gpus.append(base_gpu)
            base_gpu += ngpus_per_node
        self.base_gpus.append(base_gpu)
        links = node_links
        if copies > 1:
            links = _add_ext_edge(node_links, ngpus_per_node, self.ngpus, copies, internode_conn, 1)
            self.invbws = _add_ext_edge(node_invbws, ngpus_per_node, self.ngpus, copies, internode_conn, remote_invbw)
            if copies > 2:
                ext_switches = _add_ext_switches(self.invbws, ngpus_per_node)

        self.links = links
        self.remote_invbw = remote_invbw
        self.remote_alpha = remote_alpha

        self.switches = _make_switch(switches, self.invbws, copies, ngpus_per_node)
        if len(ext_switches) > 0:
            if len(self.switches) == 0:
                self.switches.append(ext_switches)
            else:
                self.switches[0].extend(ext_switches)
        self.num_switches = len(switches)

        # switches = [switch1, switch2, ...]
        # Have all unique switch src-dsts in switch1
        # If there are 6 switches as in nvswitch in DGX2, we separate them in switch1, switch2, ..., switch6
        # switch1 = [([src0], [dsts], 1, invbw, "out-1"), ([srcs], [dst0], 1, invbw, "in-1"),...]
        # switch2 = [([src0], [dsts], 1, invbw, "out-2"), ([srcs], [dst0], 1, invbw, "in-2"),...]
        for switch in self.switches:
            # print("switch:", switch)
            for srcs, dsts, lk, invbw, switch_name in switch:
                if lk == 0:
                    raise ValueError(f'Switch {switch_name} has zero bandwidth, but switch bandwidths must be strictly positive. Please encode connectedness in links.')
                if lk < 0:
                    raise ValueError(f'Switch {switch_name} has a negative inverse bandwidth of {invbw}. Bandwidth must be strictly positive.')
        self.bw_dist, _ = self.set_bw_distances()
        self.set_switches_involved()
        self.set_L()

    def gpu_to_node(self, gpu):
        assert gpu < self.ngpus
        assert gpu >= 0
        for c in range(self.copies):
            if gpu >= self.base_gpus[c] and gpu < self.base_gpus[c+1]:
                return c

    def node_to_gpu(self, node):
        assert node < self.copies
        assert node >= 0
        return sum(self.base_gpus[:node+1])

    def sources(self, dst):
        for src, bw in enumerate(self.links[dst]):
            if bw > 0:
                yield src

    def destinations(self, src):
        for dst, links in enumerate(self.links):
            bw = links[src]
            if bw > 0:
                yield dst

    def link(self, src, dst):
        return self.links[dst][src]

    def get_invbw(self, src, dst):
        return self.invbws[dst][src]

    def num_nodes(self):
        return len(self.links)

    def nodes(self):
        return range(self.num_nodes())

    # constraints using number of links
    def bandwidth_constraints(self):
        for dst, dst_links in enumerate(self.links):
            for src, lk in enumerate(dst_links):
                if lk > 0:
                    yield ([src], [dst], lk, f'{src}→{dst}')
        for switch in self.switches:
            for srcs, dsts,lk, _, switch_name in switch:
                yield (srcs, dsts, lk, switch_name)

    # constraints using actual bandwidth
    def real_bandwidth_constraints(self):
        for dst, dst_links in enumerate(self.invbws):
            for src, invbw in enumerate(dst_links):
                if invbw > 0:
                    for l in range(self.link(src,dst)):
                        yield ([src], [dst], invbw, l, f'{src}→{dst}')
        for swt_i, switch in enumerate(self.switches):
            for srcs, dsts, _, invbw, switch_name in switch:
                yield (srcs, dsts, invbw, swt_i, switch_name)

    def set_bw_distances(self):
        if self.remote_beta is None:
            return None, None
        # Floyd–Warshall algorithm for all-pairs shortest paths with path information
        # Modified to track all shortest paths
        nodes = range(self.num_nodes())
        dist = [[math.inf for _ in nodes] for _ in nodes]
        next = [[set() for _ in nodes] for _ in nodes]
        for dst in nodes:
            for src in self.sources(dst):
                dist[src][dst] = self.invbws[dst][src]
                next[src][dst].add(dst)
        for node in nodes:
            dist[node][node] = 0
            next[node][node].add(node)
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if dist[i][j] >= dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next[i][j].update(next[i][k])
        return dist, next

    def set_switches_involved(self):
        self.switches_involved = defaultdict(list)
        self.switches_involved_in = defaultdict(list)
        self.switch_dst_dict = defaultdict(dict)
        self.switch_src_dict = defaultdict(dict)
        for i, switch in enumerate(self.switches):
            for srcs, dsts, _, _, switch_name in switch:
                if "out" in switch_name:
                    assert len(srcs) == 1
                    self.switch_dst_dict[srcs[0]][(i,switch_name[:-6])] = dsts
                    for dst in dsts:
                        self.switches_involved[(srcs[0],dst)].append((i,switch_name[:-6]))
                if "in" in switch_name:
                    assert len(dsts) == 1
                    self.switch_src_dict[dsts[0]][(i,switch_name[:-6])] = srcs
                    for src in srcs:
                        self.switches_involved_in[(src,dsts[0])].append((i,switch_name[:-6]))
        # print("si: ", self.switches_involved)
        # print("sii: ", self.switches_involved_in)

    def reverse_links(self):
        num_nodes = self.num_nodes()
        new_links = [[None for y in range(num_nodes)] for x in range(num_nodes)]
        for x in range(num_nodes):
            for y in range(num_nodes):
                new_links[x][y] = self.links[y][x]
        new_invbws = [[None for y in range(num_nodes)] for x in range(num_nodes)]
        for x in range(num_nodes):
            for y in range(num_nodes):
                new_invbws[x][y] = self.invbws[y][x]
        new_switches = []
        for i in range(self.num_switches):
            new_swt = []
            for swt in self.switches:
                for srcs, dsts, lk, invbw, name in swt:
                    if "out" in name:
                        new_name = name.replace('out','in')
                    elif "in" in name:
                        new_name = name.replace('in','out')
                    new_swt.append((dsts,srcs,lk,invbw,new_name))
            new_switches.append(new_swt)
        
        self.links = new_links
        self.invbws = new_invbws
        self.switches = new_switches
        self.set_switches_involved()

    def set_L(self):
        R = self.num_nodes()
        L = 0
        for src in range(R):
            for dst in self.destinations(src):
                if self.link(src,dst) > L:
                    L = self.link(src,dst)
        self.L = L
