
def greedy_mvc(G):
    covered_set = set()
    numCoveredEdges = 0
    idxes = range(nx.number_of_nodes(G))
    idxes = sorted(idxes, key=lambda x: len(nx.neighbors(G, x)), reverse=True)
    pos = 0
    while numCoveredEdges < nx.number_of_edges(G):
        new_cov_node = idxes[pos]
        covered_set.add(new_cov_node)
        for neigh in nx.neighbors(G, new_cov_node):
            if neigh not in covered_set:
                numCoveredEdges += 1
        pos += 1
    print('done')
    return covered_set

