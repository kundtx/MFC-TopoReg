import numpy as np
import csv
import copy
import networkx as nx
import pickle

def save_any_obj_pkl(obj, path):
    ''' save any object to pickle file
    '''
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def read_dblp_dat_and_create_temporal_networks_accumulate_edges(file = 'dblp-2019-12-01.dat'):

    min_year = 2010
    max_year = 2019
    
    #cora 
    temporal_networks = create_empty_network(min_year, max_year)

    f = open(file)
    lines = f.readlines()
    #for i in range(len(lines)):
    dblp_label_dict = [{} for y in range(max_year-min_year+1)]
    edges = []
    for i in range(len(lines)):   # read each line of the file
        single_edge = lines[i][0:-1].split('\t')
        if i % 1000 == 0:
            print(i,'  ',len(lines))
        author_1 = single_edge[0]
        author_2 = single_edge[1]
        count = single_edge[2]
        year = int(single_edge[3])
        author_1_label = single_edge[4]
        author_2_label = single_edge[5]

        if author_1_label == '0' or author_2_label == '0': # no all in the 14 classes
            continue

        temporal_index = year-min_year
        if 0<=temporal_index < len(temporal_networks):
            temporal_networks[temporal_index].add_edge(author_1, author_2) 
            dblp_label_dict[temporal_index][author_1] = author_1_label
            dblp_label_dict[temporal_index][author_2] = author_2_label

    save_any_obj_pkl(temporal_networks, 'dblp_temporal_network.pkl')
    #final_dblp = {}
    #for i in range(len())
    save_any_obj_pkl(dblp_label_dict, 'DBLPdyn_label.pkl')
    #for i in range(len())


def create_empty_network(min_year, max_year):
    all_network = []
    for i in range(min_year, max_year+1):
        current_t = nx.Graph()
        all_network.append(current_t)
    return all_network

def check_dblp_label_consistence(file = 'dblp-2019-12-01.dat'):#done it is consistent

    author_label_dict = {}
    f = open(file)
    lines = f.readlines()
    #for i in range(len(lines)):
    edges = []

    for i in range(len(lines)):
        if i % 1000 == 0:
            print(i,'  ',len(lines))
        single_edge = lines[i][0:-1].split('\t')
        author_1 = single_edge[0]
        author_2 = single_edge[1]
        count = single_edge[2]
        year = single_edge[3]
        author_1_label = single_edge[4]
        author_2_label = single_edge[5]
        author_label_dict[author_1] = author_1_label
        author_label_dict[author_2] = author_2_label
        save_any_obj_pkl(author_label_dict, 'dblp_label_dict.pkl')



if __name__ == "__main__":
    read_dblp_dat_and_create_temporal_networks_accumulate_edges()
    #check_dblp_label_consistence()
    