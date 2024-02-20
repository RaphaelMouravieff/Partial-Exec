
# visu
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from .utils import find_neighboor, find_last_edges, find_first_edges, remove_edges_first, compare_objects




class Graph:
    def __init__(self, edges, nodes, logical_form):
         
        self.edges = edges
        self.nodes = nodes
        self.logical_form = logical_form
        self.Omega_include = None
        self.mapping_exec = None
        self.prefixs = {"P", "C", "S", "GB", "H", "A", "O", "L"}

    def execute_node(self, node, parent_counts):
        if node.node_name not in parent_counts:
            parent_counts[node.node_name] = 0

        # Execute all operands first and collect their results
        operand_results = [self.execute_node(operand, parent_counts) for operand in node.operands]

        # Execute the node with the results of its operands
        result = node.execute(operand_results)

        # Store the result in the cache
        parent_counts[node.node_name] +=1
        return result
    

    def remove_operator_comp(self):
        remove = None
        for _, node in self.nodes.items():
            if node.prefix == "C":
                if node.any_tuples == True:
                    remove, index = (node.operands[0].node_name,1) if node.operands[0].prefix == "OP" else (node.operands[1].node_name,0)
                    node.operands = [node.operands[index]]
        if remove is not None:
            self.edges = [(e1,e2) for e1,e2 in self.edges if e1!=remove]
            self.nodes.pop(remove)
            self.parent_counts.pop(remove)
                


    def executed_last_node(self):
        last_node = find_last_edges(self.edges)
        if last_node is not None:
            node = self.nodes[last_node[0]]
            parent_counts = {}
            result = self.execute_node(node, parent_counts)
            self.parent_counts = parent_counts
            self.remove_operator_comp()
        else:
            result = self.nodes[list(self.nodes.keys())[0]].execute()
        result = self.prep_answer(result)
        return result

    def prep_answer(self, result):
        try:
            result = result.explode() if hasattr(result,"explode") else result # pd.Series with list inside to pd.Series without
        except:
            pass
        result = result.tolist() if hasattr(result, 'tolist') else result # pd.Series to list
        result = [result] if type(result) in [str,int,bool, float] else result
        result = list(set(result)) if len(set(result))==1 else result # keep only one element 
        result = [str(r) for r in result] # create string
        return result

    def update_execution_status(self, Omega_include):
        self.Omega_include = Omega_include
        for node in self.nodes.values(): # By default, no node is executed
            node.is_exec = False

        edges = self.edges
        first_turn=0
        while len(edges)>0:
            first_edges = find_first_edges(edges)
            edges = remove_edges_first(edges, first_edges)
            if first_turn==0:
                for source in first_edges:
                    if self.nodes[source].prefix in Omega_include:
                        self.nodes[source].is_exec = True
                        
            if first_turn == 1:
                for source in first_edges:
                    children = self.find_parents_or_children_nodes(source, mode="child")
                    if all([self.nodes[c].prefix in Omega_include  for c in children]) and self.nodes[source].prefix in Omega_include:
                        self.nodes[source].is_exec = True
            first_turn = 1

        if len(self.nodes)==1:
            last_edge = list(self.nodes.keys())[0]
        else:
            last_edge = find_last_edges(self.edges)[0]
        children = self.find_parents_or_children_nodes(last_edge, mode="child")
        if all([self.nodes[c].prefix in Omega_include  for c in children]) and self.nodes[last_edge].prefix in Omega_include:
            self.nodes[last_edge].is_exec = True

        for _, node in self.nodes.items() : 
            if not node.is_exec:
                for operand in node.operands:
                    if operand.is_exec:
                        operand.is_exec="last"

        if all(node.is_exec for _,node in self.nodes.items()):
            self.nodes[last_edge].is_exec = "last"

        self.mapping_exec = {node_name: str(node.is_exec * 1) for node_name, node in self.nodes.items()}


    def linearize_graph(self, Omega_include, flatten_mode = "preorder"):
        assert flatten_mode in ["preorder", "postorder", "preorder_alias_start","postorder_alias_start","preorder_alias_end", "postorder_alias_end"],"\
            flatten_mode must be : 'preorder', 'postorder', 'preorder_alias_start', 'postorder_alias_start', 'preorder_alias_end', 'postorder_alias_end'"

        flatten_mode = flatten_mode.split('_')
        if len(flatten_mode)==3:
            no_repeats = True
            operand_first = True if flatten_mode[2]=="start" else False
            flatten_mode = flatten_mode[0]

        elif len(flatten_mode)==1:
            flatten_mode = flatten_mode[0]
            no_repeats = False
            operand_first = None
            

        last_node = find_last_edges(self.edges)
        self.update_execution_status(Omega_include)
        executed_nodes = {}
        if no_repeats==False:
            flatten_sequence = []

        if no_repeats==True:
            flatten_sequence_operands = []
            flatten_sequence_operators = []

        if last_node is not None and "last" in list(self.mapping_exec.values()) and "0" in list(self.mapping_exec.values()):
            if flatten_mode == "preorder":
                if no_repeats==False:
                    self.flatten_preorder(self.nodes[last_node[0]], executed_nodes, flatten_sequence)
                if no_repeats:
                    self.compare_nodes()
                    self.flatten_preorder_no_repeats(self.nodes[last_node[0]], executed_nodes, flatten_sequence_operands, flatten_sequence_operators)

            if flatten_mode == "postorder":
                if no_repeats==False:
                    self.flatten_postorder(self.nodes[last_node[0]], executed_nodes, flatten_sequence)
                if no_repeats:
                    self.compare_nodes()
                    self.flatten_postorder_no_repeats(self.nodes[last_node[0]], executed_nodes, flatten_sequence_operands, flatten_sequence_operators)

            if not no_repeats and not operand_first:
                flatten_sequence = ''.join(flatten_sequence)

            else:
                if operand_first:
                    flatten_sequence = ''.join(flatten_sequence_operands)+"|"+''.join(flatten_sequence_operators)
                if not operand_first:
                    flatten_sequence = ''.join(flatten_sequence_operators)+"|"+''.join(flatten_sequence_operands)

        else: # flatten graph with only one node
            key_of_last = next(key for key, value in self.mapping_exec.items() if value == 'last')

            flatten_sequence = self.nodes[key_of_last].linearize(mode="noalias")
        return flatten_sequence


    def flatten_preorder(self, node, executed_nodes, flatten_sequence):

        # If the node is already executed, return
        if executed_nodes.get(node.node_name, 0) == self.parent_counts.get(node.node_name, 1):
            return
        # Increment the executed count for this node
        executed_nodes[node.node_name] = executed_nodes.get(node.node_name, 0) + 1
        # If the node has been executed the required number of times, process it
        if executed_nodes[node.node_name] <= self.parent_counts[node.node_name]:
            # Activate the 'linearize' method and add its result to the flatten_sequence
            linearized_result = node.linearize(mode="noalias")
            if linearized_result is not None:
                flatten_sequence.append(linearized_result)

        reverse = False
        # Recursively call the function on operands and collect their linearized results
        if len(node.operands)==2: # determine which operand to linearize first
            if sum([i.idx for i in node.operands]): # There is a distinction in the order
                reverse = [i.idx for i in node.operands].index(True) == 1

        for operand in node.operands if reverse==False else node.operands[::-1]:
            self.flatten_preorder(operand, executed_nodes, flatten_sequence)


    def flatten_postorder(self, node, executed_nodes, flatten_sequence):
        # Check if the node has already been executed, if so, return
        if executed_nodes.get(node.node_name, 0) == self.parent_counts.get(node.node_name, 1):
            return

        # Determine the order of operands processing
        reverse = False
        if len(node.operands) == 2:  # Only apply ordering logic if there are two operands
            if sum([i.idx for i in node.operands]):  # Check for distinct order
                reverse = [i.idx for i in node.operands].index(True) == 1

        # Process each operand recursively first
        for operand in node.operands if reverse else node.operands[::-1]: # reverse 
            self.flatten_postorder(operand, executed_nodes, flatten_sequence)

        # Increment the executed count for this node
        executed_nodes[node.node_name] = executed_nodes.get(node.node_name, 0) + 1

        # Process the current node if it has been executed the required number of times
        if executed_nodes[node.node_name] <= self.parent_counts[node.node_name]:
            # Activate the 'linearize' method and add its result to the flatten_sequence
            linearized_result = node.linearize(mode="noalias")
            if linearized_result is not None:
                flatten_sequence.append(linearized_result)


    def flatten_postorder_no_repeats(self, node, executed_nodes, flatten_sequence_operands, flatten_sequence_operators):
        # Check if the node has already been executed, if so, return
        if executed_nodes.get(node.node_name, 0) == self.parent_counts.get(node.node_name, 1):
            return
        # Increment the executed count for this node
        # Determine the order of operands processing
        reverse = False
        if len(node.operands) == 2:  # Only apply ordering logic if there are two operands
            if sum([i.idx for i in node.operands]):  # Check for distinct order
                reverse = [i.idx for i in node.operands].index(True) == 1

        # Process each operand recursively first
        for operand in node.operands if reverse else node.operands[::-1]: # reverse 
            self.flatten_postorder_no_repeats(operand, executed_nodes, flatten_sequence_operands, flatten_sequence_operators)
            
        executed_nodes[node.node_name] = executed_nodes.get(node.node_name, 0) + 1
        # Process the current node if it has been executed the required number of times
        if executed_nodes[node.node_name] == 1:
                    
            # Activate the 'linearize' method and add its result to the flatten_sequence
            linearized_result = node.linearize(mode="alias")
            if linearized_result is not None:
                if node.is_exec=="last" :
                    flatten_sequence_operands.append(linearized_result)
                else:
                    flatten_sequence_operators.append(linearized_result)
                
    def flatten_preorder_no_repeats(self, node, executed_nodes, flatten_sequence_operands, flatten_sequence_operators):
        # If the node is already executed, return
        if executed_nodes.get(node.node_name, 0) == self.parent_counts.get(node.node_name, 1):
            return
        # Increment the executed count for this node
        executed_nodes[node.node_name] = executed_nodes.get(node.node_name, 0) + 1
        # If the node has been executed the required number of times, process it
        if executed_nodes[node.node_name] == 1:
                    
            # Activate the 'linearize' method and add its result to the flatten_sequence
            linearized_result = node.linearize(mode="alias")
            if linearized_result is not None:
                if node.is_exec=="last" :
                    flatten_sequence_operands.append(linearized_result)
                else:
                    flatten_sequence_operators.append(linearized_result)

        reverse = False
        # Recursively call the function on operands and collect their linearized results
        if len(node.operands)==2: # determine which operand to linearize first
            if sum([i.idx for i in node.operands]): # There is a distinction in the order
                reverse = [i.idx for i in node.operands].index(True) == 1

        for operand in node.operands if reverse==False else node.operands[::-1]:
            self.flatten_preorder_no_repeats(operand, executed_nodes, flatten_sequence_operands, flatten_sequence_operators)



            
    def find_parents_or_children_nodes(self, start_node, mode):
        graph = {}
        for edge in self.edges:
            parent, child = edge
            if mode=="parent":
                if parent not in graph:
                    graph[parent] = []
                graph[parent].append(child)
            if mode=="child":
                if child not in graph:
                    graph[child] = []
                graph[child].append(parent)
        
        def dfs(node, visited, result):
            visited.add(node)
            result.append(node)
            if node in graph:
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor, visited, result)
        visited = set()
        result = []
        dfs(start_node, visited, result)
        return result[1:]  # Exclude the start_node itself from the result


    def compare_nodes(self):
        last_nodes = [node for node, mark in self.mapping_exec.items() if mark == 'last']
        same_node_name = set()
        for i in range(len(last_nodes)):
            for j in range(i + 1, len(last_nodes)):
                node1, node2 = last_nodes[i], last_nodes[j]
                result1 = self.nodes[node1].result
                result2 = self.nodes[node2].result
                if compare_objects(result1, result2):
                    
                    Nx = self.nodes[node1].node_name
                    Ny = self.nodes[node2].node_name
                    if Nx in same_node_name:
                        self.nodes[node2].node_name = self.nodes[node1].node_name 
                    if Ny in same_node_name:
                        self.nodes[node1].node_name  = self.nodes[node2].node_name 
                    else :
                        self.nodes[node1].node_name  = self.nodes[node2].node_name 
                        same_node_name.update([Ny])


    def create_cool_dag(self, execution_status=False):

        if execution_status == True:
            assert self.mapping_exec is not None, "Update execution status with Omega_include first."
            assert self.Omega_include is not None, "Update execution status with Omega_include first."
        
        def mapping(nodes, edges):
            mapping_name = {}
            mapping_color = {}
            for edge in edges:
                for element in edge:
                    if element not in mapping_name:  # Check if element is already mapped
                        color_name = nodes[element].prefix
                        name = element+" "+nodes[element].parameter
                        mapping_name[element.split('|')[0]] = name
                        mapping_color[element.split('|')[0]] = color_name
            return mapping_name, mapping_color

        mapping_name, mapping_color = mapping(self.nodes, self.edges)

        if execution_status == True:
            mapping_color = self.mapping_exec

        dag = nx.DiGraph()

        # Prepare a color map for each category in the mapping_color
        categories = set(mapping_color.values())
        category_colors = plt.cm.Pastel1(np.linspace(0, 1, len(categories)))
        color_dict = dict(zip(categories, category_colors))
        if execution_status == True:
            color_dict = {'0': np.array([0.98431373, 0.70588235, 0.68235294, 1.]),
                        'last': np.array([0.99607843, 0.85098039, 0.65098039, 1.]),
                        '1': np.array([0.94901961, 0.94901961, 0.94901961, 1. ])}

        # Adding nodes with their respective colors and names
        for node in set([n for edge in self.edges for n in edge]):
            color_category = mapping_color.get(node, 'gray')  # Default to gray if not in mapping_color
            node_label = mapping_name.get(node, node) if mapping_name else node  # Use mapping_name if provided
            dag.add_node(node, color=color_dict.get(color_category, 'gray'), label=node_label)

        # Adding edges
        for edge in self.edges:
            dag.add_edge(*edge)

        # Assign levels to nodes based on topological sorting
        levels = {}
        for node in nx.topological_sort(dag):
            levels[node] = 0 if not list(dag.predecessors(node)) else max(levels[predecessor] for predecessor in dag.predecessors(node)) + 1
        for node, level in levels.items():
            if node == "OP0semi" and level==0:
                neighb = find_neighboor("OP0semi",self.edges)
                level = levels[neighb]
            dag.nodes[node]['layer'] = level
            
        

        # Set positions using multipartite layout
        pos = nx.multipartite_layout(dag, subset_key="layer",align='vertical',scale=2, center=(0, 0))

    # pos = nx.spring_layout(dag, k=0.5, iterations=20)  # Adjust k for spacing and iterations for layout precision
        #pos = nx.circular_layout(dag)
        
        # Drawing the graph
        plt.figure(figsize=(18, 11), facecolor='white')
        node_colors = [dag.nodes[node]['color'] for node in dag.nodes()]

        # Draw nodes and edges
        nx.draw_networkx_edges(dag, pos, edge_color='gray', arrowstyle='->', arrowsize=15)
        nx.draw_networkx_nodes(dag, pos, node_color=node_colors, node_size=2000)

        # Draw node labels based on mapping_name
        node_labels = {node: dag.nodes[node]['label'] for node in dag.nodes()}
        nx.draw_networkx_labels(dag, pos, labels=node_labels, font_size=11, font_color='black', font_weight='bold')

        # Draw arrow symbols as edge labels
        edge_labels = {edge: '→' for edge in dag.edges()}
        nx.draw_networkx_edge_labels(dag, pos, edge_labels=edge_labels, font_color='black')

        # Display the query in the upper left corner if it's provided
        if self.logical_form:
            plt.text(0.0, 1.05, self.logical_form, transform=plt.gca().transAxes, fontsize=12, color='blue')
        if execution_status == True:
            plt.text(0.0, 1.00, self.Omega_include, transform=plt.gca().transAxes, fontsize=12, color='red')
        if execution_status == True:
            legend_labels = {key: f"{key}" for key in color_dict.keys()}
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[key], 
                                    markersize=10, markerfacecolor=value) for key, value in color_dict.items()]
            plt.legend(handles=legend_elements, title="Execution status", loc="upper right")

        # Display the graph
        plt.axis('off')
        plt.show()