import numpy as np
import networkx as nx


# = Breadth-first SearchAlgorithm Implementation =

# procedure BFS(G, root) is
    # let Q be a queue
    # label root as discovered
    # Q.enqueue(root)
    # while Q is not empty do
        # v := Q.dequeue()
        # if v is the goal then
            # return v
        # for all edges from v to w in G.adjacentEdges(v) do
            # if w is not labeled as discovered then
                # label w as discovered
                # Q.enqueue(w)

def breadth_first_search(G, start, end, attribute):
    path = []
    # TO DO
    return path

def depth_first_search(G, start, end, attribute):
    path = [start]
    stack = []
    visited = [start]
    print(f'depth first search begins with {start} -> {end}')

    while start != end:
        # print(f'current city: {start}')
        is_found = False
        neighbors = G.neighbors(start)

        for neigbor in neighbors:
            # print(f' > checking neighbour {neigbor}')
            if neigbor not in visited:
                visited.append(neigbor)
                is_found = True
                break
        
        if not is_found:
            start = path[-2]
            path = path[:-1]
        else:
            start = visited[-1]
            path.append(visited[-1])

    return path

def uniform_cost_search(G, start, end, attribute):
    path = []
    path = nx.dijkstra_path(G, source=start, target=end, weight=attribute)
    return path

def greedy_search(G, start, end, attribute):
    path = [start]
    optimal_neighbor = start
    visited = []
    is_found = False
    print(f'greedy search begins with {start} -> {end}')

    i = 0
    while start != end:
        min_value = np.inf

        for neighbor in G.neighbors(start):
            data = G.get_edge_data(start, neighbor)
            if neighbor == end:
                path.append(end)
                is_found = True
                break
            
            if min_value > data[attribute] and neighbor not in path and neighbor not in visited:
                min_value = G.get_edge_data(start, neighbor)[attribute]
                optimal_neighbor = neighbor
        
        if is_found:
            break
        if min_value == np.inf:
            if len(path) < 2:
                print('not found!')
                return []
            visited.append(path[-1])
            start = path[-2]
            path = path[:-1]
            continue
        
        path.append(optimal_neighbor)
        start = optimal_neighbor
    
    print('successful match!')
    return path

def a_star_search(G, start, end, attribute):
    path = []
    path = nx.astar_path(G, source=start, target=end, weight=attribute)
    return path

def find_path(algorithm, G, start, end, attribute):
    if start is None or end is None:
        return list()

    if algorithm == 'greedy':
        return greedy_search(G, start, end, attribute)
    elif algorithm == 'depth first':
        return depth_first_search(G, start, end, attribute)
    elif algorithm == 'breadth first':
        return breadth_first_search(G, start, end, attribute)
    elif algorithm == 'a*':
        return a_star_search(G, start, end, attribute)
    
    return uniform_cost_search(G, start, end, attribute)
