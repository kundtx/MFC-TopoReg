import pickle
import numpy as np
import networkx as nx  #nx2.3 is required
import copy

def load_any_obj_pkl(path):
    ''' load any object from pickle file
    '''
    with open(path, 'rb') as f:
        any_obj = pickle.load(f)
    return any_obj

def save_any_obj_pkl(obj, path):
    ''' save any object to pickle file
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def create_dynwalks_connected_data(path, new_file_name = 'unknown.pkl'):
    original_dynamic_networks = load_any_obj_pkl(path)
    new_dynamic_networks = []
    G = original_dynamic_networks[0]
    largext_connected_component = max([G.subgraph(c).copy() for c in nx.connected_components(G)], key=len)#code from documents of networkx

    largest_cc_initial_nodes = set(largext_connected_component.nodes())
    

    new_dynamic_networks.append(largext_connected_component)
    for i in range(1, len(original_dynamic_networks)):
        print(i)
        G = original_dynamic_networks[i]
        current_components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        _tmp_graph = nx.Graph()
        for single_component in current_components:
            single_component_nodes = set(single_component.nodes())
            if len(list(largest_cc_initial_nodes.intersection(single_component_nodes))) > 0:
                _tmp_graph = nx.union(_tmp_graph, single_component) # must disjoint 
        new_dynamic_networks.append(_tmp_graph)
        
    for i in range(len(new_dynamic_networks)):
        print('new:',len(list(new_dynamic_networks[i].nodes())), '   old:',len(list(original_dynamic_networks[i].nodes())))
        print('edge:',len(list(new_dynamic_networks[i].edges())), '   edge:',len(list(original_dynamic_networks[i].edges())))

    save_any_obj_pkl(new_dynamic_networks, new_file_name)
    

    

if __name__ == "__main__":
    import time
    # time.sleep(10)
    create_dynwalks_connected_data('dblp_temporal_network.pkl', 'DBLP_dyn.pkl')
