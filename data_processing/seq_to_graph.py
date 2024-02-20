
import pandas as pd 
import numpy as np
import re

from .utils import can_convert_to_int, can_convert_to_float
from .nodes import node_classes
from .graph import Graph

def find_next_operation(idx_nodes, nodes2):
    for name in idx_nodes:
        potential_operator = nodes2[name]
        
        if potential_operator.Noperands == 2: # check if the two next elements are exec
            name1, name2 = find_next_elements(idx_nodes, name, 2)
            potential_operand1 = nodes2[name1]
            potential_operand2 = nodes2[name2]
            if potential_operand1.seq_out and potential_operand2.seq_out:
                remove = [name1, name2]
                edge = [(name1, name),(name2, name)]
                nodes2[name].seq_out = 1
                return remove, edge, nodes2
            
        if potential_operator.Noperands == 1: # check if the two next elements are exec
            name1 = find_next_elements(idx_nodes, name, 1)[0]
            potential_operand1 = nodes2[name1]
            if potential_operand1.seq_out:
                remove = [name1]
                edge = [(name1, name)]
                nodes2[name].seq_out = 1
                return remove, edge, nodes2  
            
def remove_elements(main_list, elements_to_remove):
    if not isinstance(elements_to_remove, list):
        elements_to_remove = [elements_to_remove]

    for item in elements_to_remove:
        if item in main_list:
            main_list.remove(item)

def find_next_elements(lst, item, num_elements=1):
    try:
        idx = lst.index(item)
        if idx + num_elements < len(lst):
            next_elements = lst[idx + 1 : idx + 1 + num_elements]
            if type(next_elements)!=list:
                next_elements = [next_elements]
            return next_elements
        else:
            return "Not enough elements after the specified item"
    except ValueError:
        return "Item not found in the list"
    
    
def parse_node_seq(node_seq, header):
    C = [">",">=","<=","<","=","!=","in","is","intersect","union",","]
    F = ["min","max","sum","avg","abs","julianday","count","distinct","length"]
    S = ["where","and","or"]
    O = ["+","-","/"]
    # Check for non exec nodes
    #node_seq = node_seq.strip()
    # A
    for f in F:
        if node_seq==f:
            return False, "A", f, 1, None
    # S
    for s in S:
        if node_seq==s:
            return False, "S", s, 2, None   
    
    # OP one or two operands
    for o in O:
        if node_seq==o:
            return False, "OP", node_seq, 2, None
        
    for o_ in [node_seq.split(' ')[0], node_seq.split(' ')[-1]]:
        for o in O:
            if o_ ==o:
                if "| " not in node_seq:
                    return False, "OP", node_seq, 1, None

    # C one or two operands
    for c in C:
        if node_seq==c:
            return False, "C", node_seq, 2, None
        
    for c_ in [node_seq.split(' ')[0], node_seq.split(' ')[-1]]:
        for c in C:
            if c_ == c:
                if "| " not in node_seq:
                    return False, "C", node_seq, 1, None

    # L
    if node_seq.startswith('l '):
        K = node_seq.split('l ')[1]
        if K.isnumeric():
             return False, "L", node_seq, 1, None
    # P
    if node_seq in header:
        return False, "P", node_seq, 0, None
    # GB
    if node_seq == "gb":
        return False, "GB", "gb", 2, None
    # OB
    if node_seq.startswith('ob'):
        direction = node_seq.split(' ')[1]
        if direction == "ob":
            return False, "OB", "ob", 2, None
        if direction == 'asc':
            return False, "OB", "ob asc", 2, None
        if direction == 'desc':
            return False, "OB", "ob desc", 2, None
    # H
    if node_seq in ["H","h"]:
        return False, "H", "h", 2, None
    
    else:
        return True, "P", "p", 0, prep_node_result(node_seq)

    
def prep_node_result(node_seq):
    if node_seq.strip() == "":
        return pd.Series([],dtype="object")
    result = node_seq.split('| ')
    result = [r  if r!="none" else None for r in result ]
    if sum([can_convert_to_int(r)  for r in  result])==len(result):
        return pd.Series(result).astype(int)
    #if sum([can_convert_to_bool(r)  for r in  result])==len(result):
    #    return pd.Series(result).astype(bool)
    if sum([can_convert_to_float(r)  for r in  result])==len(result):
        return pd.Series(result).astype(float)
    
    if all(item in {'t', 'f', 'none'} for item in result):
        return pd.Series(result).map({"f":False,"t":True,"none":None}).astype(bool)
    
    result = pd.Series(result)
    if result.apply(lambda x : str(x).split(',, ') if x is not None else x).explode().shape[0] > result.shape[0] :
        result = result.apply(lambda x : str(x).split(',, ') if x is not None else x)
        result = result.apply(lambda x : [] if x==[""] else x)

        if result.apply(lambda X : sum([can_convert_to_float(x)  for x in  X]) if X is not None else 1).sum() == result.explode().shape[0]:
            result = result.apply(lambda X : [float(x) for x in X] if X is not None else np.nan)
    return result


def convert_series(series):
    series = series.apply(lambda x : x if x !="none" else None)
    return series.astype(str)

def find_connections(node, connections):
    return [conn[0] for conn in connections if conn[1] == node]

def parse(flatten_sequence, header, flatten_mode="preorder"):

    assert flatten_mode in ["preorder", "postorder", "preorder_alias_start","postorder_alias_start","preorder_alias_end", "postorder_alias_end"], "mode must be either 'prefix' or 'postfix'"

    flatten_mode = flatten_mode.split('_')
    if len(flatten_mode)==3:
        no_repeats = True
        operand_first = True if flatten_mode[2]=="start" else False
        flatten_mode = flatten_mode[0]

    elif len(flatten_mode)==1:
        flatten_mode = flatten_mode[0]
        no_repeats = False
        operand_first = None


    if no_repeats==False or "|||" not in flatten_sequence:
        no_repeats = False
        parse_sequence = [z for z in flatten_sequence.split(' ||')[:-1]]
        nodes_name = [f"N{idx+1}" for idx in range(len(parse_sequence))]
        if flatten_mode == "postorder":
            parse_sequence = parse_sequence[::-1]
    if no_repeats==True and "|||"  in flatten_sequence:
        if not operand_first:
            flatten_operators, flatten_operands = flatten_sequence.split(' |||')
            parse_operators, operator_node_name = remove_and_extract(flatten_operators.split(' ||'))
            parse_operands, operand_node_name = remove_and_extract(flatten_operands.split(' ||')[:-1])
        
        if operand_first:
            flatten_operands, flatten_operators = flatten_sequence.split(' |||')
            parse_operators, operator_node_name = remove_and_extract(flatten_operators.split(' ||')[:-1])
            parse_operands, operand_node_name = remove_and_extract(flatten_operands.split(' ||'))

        parse_sequence = parse_operators + parse_operands
        nodes_name = operator_node_name + operand_node_name


        

    Nseq = len(parse_sequence)
    nodes2 = {}
    for _, (node_seq, node_name) in enumerate(zip(parse_sequence, nodes_name)):


        if Nseq!=1:
            is_exec, prefix, parameter, Noperands, result = parse_node_seq(node_seq, header)
            node = node_classes.get(prefix)(node_name, parameter, prefix)
            if is_exec:
                node.result = result

        if Nseq==1:
            node = node_classes.get("P")("N1", None, "P")
            Noperands = 1
            is_exec = True
            node.result = prep_node_result(node_seq)
        
        node.Noperands = Noperands
        node.is_exec = is_exec
        node.seq_out = 0 if node.prefix != "P" else 1
        nodes2[node_name] = node


    if Nseq==0:
        node = node_classes.get("P")("N1", None, "P")
        node.result = prep_node_result("")
        node.Noperands = 1
        node.is_exec = True
        node.seq_out = 0 if node.prefix != "P" else 1
        nodes2["N1"] = node

    idx_nodes = list(nodes2.keys())


    if no_repeats==False:
        counter = 0
        edges2 = []
        while len(idx_nodes) > 1 and counter < 100:
            remove, edge, nodes2 = find_next_operation(idx_nodes, nodes2)
            remove_elements(idx_nodes, remove)
            edges2.extend(edge)
            counter += 1

        if counter == 100:
            print("Potential problem detected: operation count reached 100.")

    if no_repeats==True:
        if not operand_first:
            parse_operators =  flatten_sequence.split(' |||')[0].split(' ||')
        if operand_first:
            parse_operators = flatten_sequence.split(' |||')[1].split(' ||')[:-1]



        edges2 = extract_edges(parse_operators)

    for n1, n2 in edges2:
        nodes2[n2].operands.append(nodes2[n1])
    #Add operands order
    for name, node in nodes2.items():
        connections = find_connections(name, edges2)
        if len(connections)==2:
            if connections[0]!=connections[1]:
                #if no_repeats==False:
                 #   print('ouiiF')
                nodes2[connections[0]].idx = 1 # first element in the sequence
                nodes2[connections[1]].idx = 0     
                #if no_repeats==True: 
                #    print('ouiiT')
                #    nodes2[connections[0]].idx = 0
                #    nodes2[connections[1]].idx = 1    

    G = Graph(edges2, nodes2, flatten_sequence)
    return G


def remove_and_extract(input_list):
    cleaned_list = []
    first_labels = []

    for string in input_list:
        # Find all node labels
        labels = re.findall(r'n\d+', string.lower())
        if labels:
            # Store the first label
            first_labels.append(labels[0].strip())
            # Remove all labels to clean the string
            cleaned_string = re.sub(r'n\d+\s*', '', string.lower()).strip()
            cleaned_list.append(cleaned_string)
    return cleaned_list, first_labels


def extract_edges(input_list):
    edges = []
    for string in input_list:
        # Convert the string to lowercase
        string = string.lower()
        # Extract the labels using regular expressions
        labels = re.findall(r'n\d+', string)
        # Assuming the first label is the source and the rest are destinations
        source = labels[0]
        destinations = labels[1:]
        # Create tuples (source, destination) for each destination
        for destination in destinations:
            edges.append((destination, source))
    return edges



