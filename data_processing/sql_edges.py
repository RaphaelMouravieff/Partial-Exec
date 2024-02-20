

from .sql_parser import parse_query
from .utils import  checks, find_last_edges, find_neighboor
import random

def find_matching_string(input_string, string_list, condition):
    for s in string_list:
        if s.startswith(input_string) and s.split("|")[-2]== condition:
            return s
    return None 

def find_matching_strings(name, keys_to_check, string_list, condition):
    out = []
    for s0 in keys_to_check:
        for s1 in string_list:
            if s1.startswith(s0)  and s1.split("|")[-2]== condition:
                out.append((name, s1) )
    return out 



def find_specific_key(expressions_keys,ls_, name):
    name2 = "|".join(name.split('|')[:-2])
    if name2 in ['L|limit|0',"A|selectabs|0","H|having|0","OP|select|0","A|orderdistinct|0"]:
        return None
    
    
    dict_keys = ["|".join(i.split('|')[:-2]) for i in expressions_keys]
    condi = name.split('|')[-2]
    keys_to_check = checks[name2]

    if name2=="C|where|1":
        if "S|whereand1|0" in dict_keys and "S|whereand2|0" in dict_keys and "S|whereaggor*|0" not in dict_keys and "C|where|3" not in dict_keys:
            out = find_matching_strings(name, ["S|whereand1|0","S|whereand2|0"], ls_, condi)
            return out

    if name2 == "C|where|0":
        if "S|whereand1|0" not in dict_keys and "S|whereor1|0" not in dict_keys and "S|whereaggor*|0" not in dict_keys:
            out = find_matching_strings(name, ["S|selectwherep0|2","S|havingwhereo|0","S|havingwhereh|0","S|selectwherep0|0","S|selectwherep1|0",
                                               "S|selectwherep0|1","S|orderwhereo|0",
                                               "S|orderwhereo|1","S|groupwheregb|0"], ls_, condi)
            return out
        
    if name2 in ["S|whereand1|0","S|whereor1|0"] and "S|whereaggand*|0" not in dict_keys and "S|whereaggor*|0" not in dict_keys:
        out = find_matching_strings(name, ["S|havingwhereh|0","S|selectwherep0|0",
                                           "S|selectwherep0|1","S|orderwhereo|0",
                                           "S|orderwhereo|1","S|groupwheregb|0"], ls_, condi)
        return out
    
    if name2 =="C|wheresemi|0" and "S|wheresemiand1|0" not in dict_keys:
        out = find_matching_strings(name, ["S|selectsemiwheresemip0|0",
                                           "S|selectsemiwheresemip1|0",
                                           "S|selectsemiwheresemip1|1",
                                               "S|ordersemiwheresemio|0"], ls_, condi)
        return out
    
    if name2 == "S|wheresemiand1|0":
        out = find_matching_strings(name, ["S|selectsemiwheresemip0|0", "S|ordersemiwheresemio|0",
                                           "S|selectsemiwheresemip1|0"], ls_, condi)
        return out
    
    if name2 == "S|groupwheregb|0" :
        out = find_matching_strings(name, ["GB|group|0","GB|order|0","GB|having|0","GB|having|1"], ls_, condi)
        return out
    
    if name2 == "P|group|0" and "S|groupwheregb|0" not in dict_keys:
        out = find_matching_strings(name, ["GB|group|0","GB|order|0","GB|having|0","GB|having|1"], ls_, condi)
        return out

    if name2 == "P|order|0" and "OB|order|0" in dict_keys and "OB|order|1" in dict_keys:
        out = find_matching_strings(name, ["OB|order|0","OB|order|1"], ls_, condi)
        return out

    for key in keys_to_check:
        if key in dict_keys:
            end = find_matching_string(key, ls_, condi)
            return [(name, end)]
                            
                            


def return_connected(expressions_global):
    result = []

    if 'Absglobal' in expressions_global:
        result.append(('last_edges', 'Absglobal'))

    if 'Oglobalend' in expressions_global:
        # If Absglobal is also in the dictionary, link it to Oglobalend
        if 'Absglobal' in expressions_global:
            result.append(('Absglobal', 'Oglobalend'))
        else:
            result.append(('last_edges', 'Oglobalend'))

    return result


def connect_global(last_edge, expressions_global):
    elements = ["OP|global|","C|global", "A|globalabs","A|global|0", "OP|global_end"]
    connections = []
    previous_element = last_edge

    for element in elements:
        for k,v in expressions_global.items():
            if k.startswith(element):
                connections.append((previous_element, k))
                previous_element = k

    return connections




def connect_elements(expressions, expressions_global):
    edges = []
    ls_ = list(expressions.keys())
    for nexp, (k,v) in enumerate(expressions.items()):

        edge = find_specific_key(expressions,ls_, k)
        
        if edge is not None:
            edges.extend(edge)
            
    if nexp>=1:
        last_edges = find_last_edges(edges)
        for e in last_edges:
            new_element = connect_global(e, expressions_global)
            for e in new_element:
                if e not in edges:
                    edges.extend([e])
    
    return edges


def labeled_edges(edges):
    labels = [f"N{i}" for i in range(1, 39)] # 39 max number of nodes in a sql query. 
    random.shuffle(labels)

    node_labels = {}
    labeled_edges = []
    node_counter = 1
    for edge in edges:
        source, target = edge

        if source not in node_labels:
            label = labels.pop()
            node_labels[source] = f"{label}|{source}"
            node_counter += 1

        if target not in node_labels:
            label = labels.pop()
            node_labels[target] = f"{label}|{target}"
            node_counter += 1

        labeled_edges.append((node_labels[source], node_labels[target]))
    return labeled_edges


def create_edges(query):
    condi_expressions, expressions_global = parse_query(query)
    edges = connect_elements(condi_expressions, expressions_global)
    condi_expressions.update(expressions_global)
    edges = labeled_edges(edges)
    return edges, condi_expressions