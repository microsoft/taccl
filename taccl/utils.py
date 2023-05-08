# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import defaultdict
import enum
import pickle as pkl
import sys

# Creating new datastructures from order information
# Order recvs coming into and going out from a GPU connected to a switch
# Input:
#   switch_chunk_order_recv[r][ll] : [(c1,src1), ...] in order
#   switch_chunk_order_send[r][ll] : [(c1,dst1), ...] in order
# Output:
#   recv_right_after[r][l][(c,srci)] = (o,srcj) => GPU r over switch l receives o from srcj right after receiving c from srci
#   send_right_after[r][l][(c,dsti)] = (o,dstj) => GPU r over switch l sends o to dstj right after sending c to dsti
#   (c,o,r,l,srci,srcj) \in recv_first_set_1 => c is recvd on r from srci anytime before o is recvd on r from srcj
#   (c,o,r,l,dsti,dstj) \in send_first_set_1 => c is sent from r to dsti anytime before o is sent from r to dstj
def add_switch_order(switch_chunk_order_recv, switch_chunk_order_send, switch_link_mapping_recv, switch_link_mapping_send, R, LL):
    recv_right_after = [[defaultdict() for l in range(LL)] for r in range(R)]
    send_right_after = [[defaultdict() for l in range(LL)] for r in range(R)]
    recv_first_set_1 = set()
    send_first_set_1 = set()

    if switch_chunk_order_recv is not None:
        assert len(switch_chunk_order_recv) == R
        for r in range(R):
            for ll, chunk_order_recv in enumerate(switch_chunk_order_recv[r]):
                for i, (c,srci) in enumerate(chunk_order_recv):
                    l = switch_link_mapping_recv[r][ll]
                    j = i + 1
                    has_after = False
                    while j<len(chunk_order_recv):
                        o, srcj = chunk_order_recv[j]
                        if srci != srcj:
                            recv_first_set_1.add((c,o,r,l,srci,srcj))
                            if not has_after:
                                recv_right_after[r][ll][(c,srci)] = (o,srcj)
                                has_after = True
                        j = j + 1
                    if not has_after:
                        recv_right_after[r][ll][(c,srci)] = (-1,-1)
                    i = i + 1
            for ll, chunk_order_send in enumerate(switch_chunk_order_send[r]):
                for i, (c,dsti) in enumerate(chunk_order_send):
                    l = switch_link_mapping_send[r][ll]
                    j = i + 1
                    has_after = False
                    while j<len(chunk_order_send):
                        o, dstj = chunk_order_send[j]
                        if dsti != dstj:
                            send_first_set_1.add((c,o,r,l,dsti,dstj))
                            if not has_after:
                                send_right_after[r][ll][(c,dsti)] = (o,dstj)
                                has_after = True
                        j = j + 1
                    if not has_after:
                        send_right_after[r][ll][(c,dsti)] = (-1,-1)
                    i = i + 1
    return recv_right_after, recv_first_set_1, send_right_after, send_first_set_1
