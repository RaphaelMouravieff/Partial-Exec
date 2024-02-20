
from sqlglot import exp
import numpy as np
import pandas as pd


from datasets import load_dataset, load_from_disk, concatenate_datasets
import json


models_d = {'tapex'  : "microsoft/tapex-large-finetuned-wtq",
            "omnitab": "neulab/omnitab-large-finetuned-wtq",
            "tapas"  : "google/tapas-large-finetuned-wtq",

            "tapex-large_p":"/home/raphael.gervillie/sql_graph/models/preorder/fine-tuned/tapex-large_p/checkpoint-4000",
            "tapex-large_pc":"/home/raphael.gervillie/sql_graph/models/preorder/fine-tuned/tapex-large_pc/checkpoint-8000",
            "tapex-large_pcs":"/home/raphael.gervillie/sql_graph/models/preorder/fine-tuned/tapex-large_pcs/checkpoint-4000",
            
            "tapex-large_pcsgbh":"/home/raphael.gervillie/sql_graph/models/preorder/fine-tuned/tapex-large_pcsgbh/checkpoint-8000",
            "tapex-large_pcsgbhob":"/home/raphael.gervillie/sql_graph/models/preorder/fine-tuned/tapex-large_pcsgbhob/checkpoint-4000",
            "tapex-large_pcsgbhoba":"/home/raphael.gervillie/sql_graph/models/preorder/fine-tuned/tapex-large_pcsgbhoba/checkpoint-6000",
            "tapex-large_pcsgbhobaop":"/home/raphael.gervillie/sql_graph/models/preorder/fine-tuned/tapex-large_pcsgbhobaop/checkpoint-2000",
            }

models_d2 = {'tapex'  : "microsoft/tapex-large-finetuned-wtq",
            "omnitab": "neulab/omnitab-large-finetuned-wtq",
            "tapas"  : "google/tapas-large-finetuned-wtq",
            "tapex-large_p":"/home/raphael.gervillie/sql_graph/models/preorder_alias_end/fine-tuned/tapex-large_p/checkpoint-6000",
            "tapex-large_pc":"/home/raphael.gervillie/sql_graph/models/preorder_alias_end/fine-tuned/tapex-large_pc/checkpoint-8000",
            "tapex-large_pcs":"/home/raphael.gervillie/sql_graph/models/preorder_alias_end/fine-tuned/tapex-large_pcs/checkpoint-8000",
            "tapex-large_pcsgbh":"/home/raphael.gervillie/sql_graph/models/preorder_alias_end/fine-tuned/tapex-large_pcsgbh/checkpoint-6000",
            "tapex-large_pcsgbhob":"/home/raphael.gervillie/sql_graph/models/preorder_alias_end/fine-tuned/tapex-large_pcsgbhob/checkpoint-4000",
            "tapex-large_pcsgbhoba":"/home/raphael.gervillie/sql_graph/models/preorder_alias_end/fine-tuned/tapex-large_pcsgbhoba/checkpoint-8000",
            "tapex-large_pcsgbhobaop":"/home/raphael.gervillie/sql_graph/models/preorder_alias_end/fine-tuned/tapex-large_pcsgbhobaop/checkpoint-6000",
            }

def custom_set_operation(v1, v2, t1, t2):
    # Convert lists to sets
    set_v1 = set(v1)
    set_v2 = set(v2)
    set_t1 = set(t1)
    set_t2 = set(t2)
    union_v = set_v1.union(set_v2)
    union_t = set_t1.union(set_t2)
    result = union_v.difference(union_t)
    return result

def common_dataset():
    file_path = "/home/raphael.gervillie/sql_graph/data/squall.json"
    with open(file_path, 'r') as json_file:
        squall = json.load(json_file)

    wtq_ours = load_from_disk('/home/raphael.gervillie/sql_graph/data/wtq_lf_preorder')
    wtq = load_dataset('wikitablequestions')
    v1 = wtq_ours["validation"]["id"]
    v2 = wtq["validation"]["id"]
    t1 = wtq_ours["train"]["id"]
    t2 = wtq["train"]["id"]
    val_id = custom_set_operation(v1,v2,t1,t2)
    print(f"val_id : {len(val_id)}")
    val_id = val_id.intersection(set([i["nt"] for i in squall]))
    print(f"val_id : {len(val_id)}")
    concat_dataset = concatenate_datasets([wtq["train"], wtq["validation"]])
    wtq_validation_indices = [i for i, example in enumerate(concat_dataset) if example['id'] in val_id]
    validation_dataset = concat_dataset.select(wtq_validation_indices)
    return validation_dataset


def init_counts():
    node_counts = {str(i):0 for i in range(19)}
    for i in range(1, 19):
        node_counts[str(i)] = 0
        node_counts[f"{i}_total"] = 0

    operation_counts = {"P": 0, "C": 0, "S": 0, "GB": 0, "H": 0, "OB": 0, "A": 0, "OP": 0, "L": 0,
                        "P_total": 0, "C_total": 0, "S_total": 0, "GB_total": 0,
                          "H_total": 0, "OB_total": 0, "A_total": 0, "OP_total": 0, "L_total": 0}
    
    return node_counts, operation_counts








def to_pandas(item):
    return pd.DataFrame(item['table']["rows"],columns=item['table']["header"])

def convert_to_numeric(sequence):
    converted_sequence = []
    all_converted = True  # Assume all can be converted initially
    
    for item in sequence:
        # Handle None and empty string as np.nan
        if item is None or item == '':
            converted_sequence.append(np.nan)
        else:
            try:
                # Try converting to integer
                converted_sequence.append(int(item))
            except ValueError:
                try:
                    # If integer conversion fails, try converting to float
                    converted_sequence.append(float(item))
                except ValueError:
                    # If conversion fails, mark as not fully converted and break
                    all_converted = False
                    break  # Early exit if any conversion fails
    
    # If not all items were converted, return original sequence and False
    if not all_converted:
        return sequence, False
    else:
        return converted_sequence, True

def compare_objects(obj1, obj2, tolerance=1e-5):
    if type(obj1) != type(obj2):
        return False
    if isinstance(obj1, (pd.Series, pd.DataFrame)):
        return obj1.equals(obj2)
    elif isinstance(obj1, list):
        if len(obj1) != len(obj2):
            return False
        return all(compare_objects(sub_obj1, sub_obj2, tolerance) for sub_obj1, sub_obj2 in zip(obj1, obj2))
    elif isinstance(obj1, dict):
        if obj1.keys() != obj2.keys():
            return False
        return all(compare_objects(obj1[key], obj2[key], tolerance) for key in obj1)
    elif isinstance(obj1, set):
        return obj1 == obj2
    elif isinstance(obj1, (int, float)):
        return abs(obj1 - obj2) <= tolerance
    else:
        return obj1 == obj2
    
def find_neighboor(node_name, edges):
    for z in edges:
        if z[0] == node_name:
            return z[1]
        
def initialize_counts_dict(nodes_dict):
    # Initialize an empty dictionary
    counts_dict = {}
    # For each key in nodes_dict, create a list of zeros of the same length as the value list
    for key in nodes_dict:
        counts_dict[key] = [0] * len(nodes_dict[key])

    return counts_dict


def convert_to_appropriate_type(value_str):
    try:
        # Try converting to an integer
        result = int(value_str)
    except ValueError:
        try:
            # Try converting to a float
            result = float(value_str)
        except ValueError:
            # If not an int or float, keep it as a string
            result = value_str
    return result

def convert_to_numeric(item, ls=True):
    if ls:
        numeric_values = []
        for val in item:
            try:
                numeric_val = float(val)
            except (ValueError, TypeError):
                numeric_val = np.nan
            numeric_values.append(numeric_val)
        return numeric_values
    if ls==False:
        try:
            numeric_val = float(item)
        except (ValueError, TypeError):
            numeric_val = np.nan
        return numeric_val

        
def remove_brackets(word):
    if (word[0] == word[-1] == "'") or (word[0] == word[-1] == '"'):
        word = word[1:-1]
    return word


def find_last_edges(graph):
    if len(graph)==0:
        return None
    first_elements = set(edge[0] for edge in graph)
    second_elements = set(edge[1] for edge in graph)
    last_edges = second_elements - first_elements
    return list(last_edges)


def find_first_edges(graph):
    if len(graph)==0:
        return None
    first_elements = set(edge[0] for edge in graph)
    second_elements = set(edge[1] for edge in graph)
    first_edges = first_elements - second_elements 
    return list(first_edges)

def remove_edges_first(edges, first_edges):
    edges = [(n1,n2) for n1,n2 in edges if n1 not in first_edges]
    return edges

replaces_sql_symbols = {"EQ":"=","GTE": ">=","LTE": "<=","NEQ": "!=",
                            "GT":">","LT": "<","Add":"+","Sub":"-","Count":"count",
                           'Distinct':"distinct","Abs":"abs","L":"limit",
                           "Anonymous":"julianday","Max":"max","Min":"min","Sum":"sum",
                            "Avg":"avg","In":"in","Is":"is",
                            "Intersect":"intersect","Union":"union","Length":"length","Div":"/"}


def translate(value):
    if type(value)==str:
        return value.lower()
    if type(value)==tuple:
        try:
            v, s = value
            s = replaces_sql_symbols[s.__name__]
        except:
            s, v = value
            s = replaces_sql_symbols[s.__name__]
        if s == "in":
            v = ", ".join([vv.sql() for vv in v])
            return f"{s} {v}".lower()
        if type(v)!=str:
            return f"{s} {v.sql()}".lower() if type(v)!= list else f"{s}".lower()
        if type(v)==str: 
            return f"{s} {v}".lower() if type(v)!= list else f"{s}".lower()
    
    if type(value) in [type(exp.Sub),type(exp.GTE),type(exp.LTE),
                             type(exp.NEQ),type(exp.GT),type(exp.LT),type(exp.Sub),
                             type(exp.Add),type(exp.EQ)]:
        return replaces_sql_symbols[value.__name__].lower()
    else:
        return value.sql().lower()

def can_convert_to_int(val):
    val_str = str(val)
    # Check if the string is just '0'
    try : 
        if val_str == '0':
            return True

        # Check for leading zeros in positive and negative numbers
        if val_str.startswith('0') or (val_str.startswith('-') and val_str[1] == '0'):
            return False
    except:
        pass

    try:
        int(val)
        return True
    except (ValueError, TypeError):
        return False
    
def can_convert_to_float(val):
    val_str = str(val)
    if val_str in ['0',"0.0"]:
        return True
    
    for start in [f"0{idx}" for idx in range(10)]:
        if val_str.startswith(start):
            return False
    
    if (val_str.startswith('-0') )and (val_str.startswith('-0.')==False):
        return False


    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False
    

def can_convert_to_bool(val):
        val_str = str(val).lower()
        if val_str in ['true', 'false', '1', '0', 'nan', 'none']:
            return True
        return False

checks = {
        "P|group|0" : ["S|groupwheregb|0", "GB|group|0"],
        "P|groupsemi|0" : ["GB|group|0"],
        "P|having0|0" : ["S|havingwhereo|0","GB|having|0"],
        "P|having0|1" : ["GB|having|1"],
        "P|select|0":["S|selectwherep0|0","GB|group|0","A|selectdistinct|0","A|select|0","OP|select0|0",'OB|order|0', "A|select*|0"],
        "P|select|1": ["S|selectwherep0|1",'OB|order|1',"A|select|1","OP|select0|0"],
        "P|select|2":["S|selectwherep0|2"],
        "P|where0|0" : ["OP|where0|0","C|where|0"],
        "P|where0|1" : ["OP|where0|0"],
        "P|wherec0|0" : ["C|where|0"],
        "P|wherec1|1" : ["C|where|1"],
        "P|where1|0" : ["OP|where0|1","C|where|1"],
        "P|where2|0" : ["C|where|2"],
        "P|where3|0" : ["C|where|3"],
        "P|where1|1" :["OP|where0|1"],
        "P|wheresemi0|0" : ["C|wheresemi|0"],
        "P|wheresemic0|0" : ["C|wheresemi|0"],
        "P|wheresemi1|0" : ["C|wheresemi|1"],
        "P|selectsemi|0" : ["S|selectsemiwheresemip1|0","S|selectsemiwheresemip0|0","OB|ordersemi|0","A|selectsemi*|0"],
        'P|selectsemi|1' : ["S|selectsemiwheresemip1|1"],
        "P|order|0":["S|orderwhereo|0","GB|order|0","A|order|0","OP|order0|0","A|orderabsabs|0","A|order*|0","OB|order|0"],
        "P|order|1":["S|orderwhereo|1","GB|order|1","A|order|1","OP|order0|0"],
        "P|ordersemi|0" : ["S|ordersemiwheresemio|0","OB|ordersemi|0"],

        "GB|group|0" : ["H|having|0", "OB|order|0"],
        "GB|groupsemi|0" : ["OB|ordersemi|0"],
        "GB|order|0" : ["A|order*|0"],
        "GB|order|1" : ["OB|order|0"],
        "GB|ordersemi|0" : ["A|ordersemi*|0"],
        "GB|having|0" : ["A|having0|0"],
        "GB|having|1" : ["A|having0|1"],

        "A|select|0":["C|select|0",'OP|select0|0','OP|select|0'],
        "A|select|1":["OP|select0|0"],
        "A|select*|0":["L|limit|0"],
        "A|order*|0":["OP|order0|0","OB|order|0",'GB|order|0'],
        'A|whereabsabs|0':["C|where|1","C|where|0"],
        "A|selectdistinct|0" : ["A|select*|0"],
        "A|selectsemi*|0" : ["OP|where|0","C|having|0","C|where|0"],
        "A|order|0" : ["OP|order0|0"],
        "A|ordersemi*|0" : ["OB|ordersemi|0"],
        "A|orderabsabs|0" : ["OB|order|0"],
        "A|order|1": ["OP|order0|0"],
        "A|having0|0" : ["OP|having0|0","C|having|0"],
        "A|having1|0" : ["C|having|1"],
        "A|having0|1" : ["OP|having0|0"],
        "A|selectabsabs|0":["OB|order|0"],
        "A|select1absabs|0":["A|select*|0"],

        "OP|having0|0" : ["C|having|0"],        
        "OP|where|0" : ["C|where|0"],
        "OP|where0|0": ['A|whereabsabs|0',"C|where|0"],
        "OP|where0|1" : ['A|whereabsabs|0', "C|where|1"],
        "OP|select0|0":["A|select1absabs|0","OP|select1|0","A|select*|0","A|selectabsabs|0"],
        "OP|selectsemi0|0" : ["C|where|1"],
        "OP|order0|0":["A|orderabsabs|0","OB|order|0"],

        "S|selectwherep0|0" : ["A|selectdistinct|0","A|select*|0","A|select|0","OP|select0|0","OP|select|0",'A|whereabsabs|0',"GB|group|0",'OB|order|0',"C|select|0",'L|limit|0'],
        "S|selectwherep0|1": ["A|select|1","OP|select0|0",'OB|order|0',"C|select|0"],
        "S|selectwherep0|2": ["OP|select1|0"],
        "S|havingwhereo|0" : ["GB|having|0"],
        "S|selectsemiwheresemip0|0": ["OB|ordersemi|0","A|selectsemi*|0","OP|where|0","C|where|0"],
        "S|selectsemiwheresemip1|0": ["OP|selectsemi0|0","OB|ordersemi|0","OP|where|0","C|where|1"],
        "S|selectsemiwheresemip1|1" : ["OP|selectsemi0|0"],
        "S|ordersemiwheresemio|0" : ["OB|ordersemi|0"],
        "S|orderwhereo|0" : ["GB|order|0","A|order|0","OP|order0|0", "OB|order|0"],
        "S|orderwhereo|1" : ["A|order|1","OP|order0|0"],
        "S|groupwheregb|0" : ["GB|order|0"],
        "S|whereand1|0":['S|whereaggor*|0','S|whereaggand*|0',"S|selectwherep0|0","S|selectwherep0|1"],
        "S|whereand2|0" : ['S|whereaggor*|0', 'S|whereaggand*|0'],
        'S|whereaggand*|0' : ["S|selectwherep0|0"],
        'S|whereaggor*|0':["S|selectwherep0|0"],
        "S|whereor1|0" :["S|selectwherep0|0"],
        "S|wheresemiand1|0":["S|selectsemiwheresemip1|0"],

        "C|wheresemi|0" : ["S|wheresemiand1|0","S|selectsemiwheresemip0|0","S|selectsemiwheresemip1|0"],
        "C|wheresemi|1" : ["S|wheresemiand1|0"],
        "C|select|0" : ["L|limit|0"],
        "C|having|0" : ["H|havingAND1|0", "H|having|0"],
        "C|having|1" : ["H|havingAND1|0"],
        "C|where|0" : ["S|whereor1|0","S|whereand1|0","S|selectwherep0|0"],
        "C|where|1" : ["S|whereand1|0","S|whereor1|0"],
        "C|where|2" : ["S|whereand2|0"],
        "C|where|3" : ["S|whereand2|0"],

        'OB|order|0':["C|select|0",'L|limit|0', "A|selectabsabs|0"],
        'OB|order|1':["C|select|0"],
        'OB|ordersemi|0' : ["L|limitsemi|0"],
        
        "H|havingAND1|0": ["H|having|0"],

        "L|limitsemi|0" : ["OP|where|0","C|where|0"],
        }

counts_dict = initialize_counts_dict(checks)

