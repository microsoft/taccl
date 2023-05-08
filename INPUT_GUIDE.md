# Writing your own topology profiles
## How to provide profiling information for topology?
### Known topologies
For any "\<topo\>" other than "custom", the link connection matrix is already defined in `./taccl/topologies/generic.py` and the topo-file only needs to have the node-config and link-profile information.

Node config information would include the following:
- `name`: the id you want to give to algorithm
- `gpus_per_node`: number of GPUs in one node
- `nics_per_node`: number of NICs in a node

Link profile would include the following:
- `alpha`: alpha-cost of the intra-node links
- `node_betas_list`: list of beta-cost of the intra-node links in an increasing order (we use a list because there can be multiple types of links within a node too, like in NVIDIA DGX-1 nodes)
- `node_invbws_list`: list of total cost (alpha + beta) of the intra-node links
- `remote_alpha`: alpha-cost of an inter-node link
- `remote_beta`: beta-cost of an inter-node link
- `remote_invbw`: remote_alpha + remote_beta

Guidelines for providing link profiles:
1. Beta values are obtained by multiplying a beta (in us/MB) by the input size (MB) for which you are trying to generate an algorithm.
2. Please ensure that all link profile costs have a big-enough integral part, since TACCL's ILP encoding will be rounding down the costs to integers in some stages and we would not want to lose information of the costs. For example, if your profile tuple is (alpha=0.3, node_betas_list=[0.5], node_invbws_list=[0.8], remote_alpha=1.5, remote_beta=2, remote_invbw=3.5), then you should multiply all values by some small factor (like 10), so that intra-node link costs don't all become 0. This will not change the synthesis problem.

### Custom topologies
In case the node topology is different from the topologies provided in the KnownTopologies class, you can set \<topo\> as "custom" and provide a "links", "betas", and "invbws" matrix instead of the list of values "node_betas_list" and "node_invbws_list" in the topology-file input.