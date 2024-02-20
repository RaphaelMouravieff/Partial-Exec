
from pandas.core.series import Series
from datetime import datetime
import numpy as np
import pandas as pd
import re


from .graph import Graph
from .utils import (remove_brackets,
                    convert_to_appropriate_type,
                    convert_to_numeric,
                    translate,
                    can_convert_to_int)

from .input_processor import InputProcessor
from .sql_edges import create_edges

def julianday(date_str):
    try:
        # Convert the date string to a datetime object
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Calculate Julian day
        year = date_obj.year
        month = date_obj.month
        day = date_obj.day
        A = year // 100
        B = 2 - A + (A // 4)
        julian_day = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
        
        return julian_day
    except ValueError:
        # Handle invalid date strings or other exceptions
        return None
    
class Node: 
    def __init__(self, node_name, parameter, prefix, df =None, position=None, condition=None, index=None, mode=None, is_exec=False, R=None, idx=None):
        self.node_name = node_name
        self.parameter = parameter
        self.df = df
        self.prefix = prefix
        self.position = position
        self.condition = condition
        self.index = index
        self.is_exec = is_exec
        self.R = R
        self.idx = idx
        self.mode = mode

    def linearize(self, mode):
        if mode=="noalias":
            if self.is_exec=="last":
                linearize_result = self.flatten_result(self.result)
                return f"{linearize_result} ||"
            if self.is_exec:
                return ""
            if not self.is_exec:
                return f"{self.parameter} ||"
            
        if mode=="alias":
            if self.is_exec=="last":
                linearize_result = self.flatten_result(self.result)
                return f"{self.node_name} {linearize_result} ||"
            if self.is_exec:
                return ""
            if not self.is_exec:
                if len(self.operands)!=2:
                    return f"{self.node_name} {self.parameter} {' '.join([operand.node_name for operand in self.operands])} ||"
                if len(self.operands)==2:
                    if self.operands[0].idx==1:
                        return f"{self.node_name} {self.parameter} {' '.join([operand.node_name for operand in self.operands])} ||"
                    else:
                        return f"{self.node_name} {self.parameter} {' '.join([operand.node_name for operand in self.operands[::-1]])} ||"
        else:
            raise ValueError(f"Node must be executed before linearizing.")
        
    def flatten_result(self, result):
        if type(result) in [float, int,str, np.float64, np.int64, bool, np.bool_]:
            flatt = str(result)
        elif isinstance(result, Series) :
            is_grouped = all(isinstance(item, list) for item in result if item is not None)
            if not is_grouped:
                if result.dtype==bool:
                    result = result.map({True:'t',False:'f'})
                else:
                    result = convert_series_to_int(result)

                flatt = "| ".join([str(x) for x in result.tolist()])
            if is_grouped:
                if all(isinstance(item, list) for item in result.explode() if item is not None):
                    result = result.apply(lambda x : [item for sublist in x for item in sublist] if x is not None else None)

                result = result.apply(lambda X : [str(x) for x in X] if X is not None else ["none"]) # for None
                flatt = "| ".join(result.apply(lambda X : ",, ".join(X)).tolist())
        elif  isinstance(result, np.ndarray):
            flatt = "| ".join([str(x) for x in result.tolist()])

        return flatt.lower()
    
    def execute(self, operand_results=None):
        raise NotImplementedError("Execute method not implemented")



def convert_series_to_int(series):
    try:
        if all(x == int(x) for x in series):
            return series.astype('Int64')  # Convert to Int64 to handle None as NaN
        else:
            return series        
    except:
        return series 
    

class Projection(Node):
    def __init__(self, *args):
        super().__init__(*args)
        self.operands = []  # Ensure operands attri

        
    def execute(self, operand_results=None):
        
        if self.is_exec==False and self.mode!="wikisql":
            self.R = self.df.loc[self.df['agg'] == 0, self.parameter]
            self.result = self.R  # Store the result

        if self.is_exec==False and self.mode=="wikisql":
            self.R = self.df.loc[:, self.parameter]
            self.result = self.R  # Store the result

        return self.result
    
            
class Comparison(Node):
    def __init__(self, *args):
        super().__init__(*args)
        self.operands = []  # Specific to ProjectionNodeA
        self.any_tuples = False

    def execute(self, operand_results = None):
        operation_mapping = {
            "=": self.EQ,
            "!=": self.NEQ,
            "<": self.LT,
            ">": self.GT,
            "<=": self.LTE,
            ">=": self.GTE,
            "in": self.In,
            "is": self.Is,
            "intersect" : self.Intersect,
            "union" : self.Union,
            "," : self.Concat
        }
        
        if self.is_exec == False: 
            self.any_tuples = any([type(i) == tuple for i in operand_results]) # indicating operation on v
            self.Noperands = len(operand_results)
            self.extract(operand_results)

            if isinstance(self.R, np.int64) or isinstance(self.R, int):
                self.is_grouped = False
            elif hasattr(self.R, '__iter__'):
                self.is_grouped = all(isinstance(item, list) for item in self.R if item is not None)
            else:
                self.is_grouped = False

            self.result = operation_mapping[self.c](self.R, self.v)


        return self.result
        
    def extract(self,operand_results):
        if self.Noperands==2 and not self.any_tuples:
            c = self.parameter

            try :
                check_ = sum([i.idx for i in self.operands])
            except:
                check_ = False
                
            if check_:
                try:
                    index = [i.idx for i in self.operands].index(True)
                except:
                    index = 1
 
                R = operand_results[index]
                v = operand_results[1] if index == 0 else operand_results[0]
                if type(R)==Series and type(v)==Series:
                    if R.shape[0]!=v.shape[0] and c  not in ["intersect","union"]:
                        v = v.iloc[0]

            elif isinstance(operand_results[0], (float, int)):
                v, R = operand_results[0], operand_results[1]
                self.operands[1].idx = 1
                self.operands[0].idx = 0
            elif isinstance(operand_results[1], (float, int)):
                R, v = operand_results[0], operand_results[1]
                self.operands[0].idx = 1
                self.operands[1].idx = 0

            elif type(operand_results[0])==Series and type(operand_results[1])==Series:
                R, v = operand_results if operand_results[0].count() >= operand_results[1].count() else operand_results[::-1]
                if R.shape[0]!=v.shape[0] and c  not in ["intersect","union"]:
                    v = v.iloc[0]
                self.operands[0].idx = 1
                self.operands[1].idx = 0

            elif not hasattr(operand_results[1],"count"):
                R,v = operand_results
                self.operands[0].idx = 1
                self.operands[1].idx = 0

            elif not hasattr(operand_results[0],"count"):
                v,R = operand_results
                self.operands[1].idx = 1
                self.operands[0].idx = 0

            else:
                R,v = operand_results
                self.operands[0].idx = 1
                self.operands[1].idx = 0

            
            
                    
        if len(operand_results)==1 or self.any_tuples:
            if self.any_tuples:
                o = operand_results[0] if type(operand_results[0])==tuple else operand_results[1]
                R = operand_results[1] if type(operand_results[1])!=tuple else operand_results[0]
            if not self.any_tuples:
                R = operand_results[0]

            c = self.parameter.split(' ')[0]
            v = " ".join(self.parameter.split(' ')[1:])

            if c!="in" and c != "is":
                if self.any_tuples:

                    v = R.dtypes.type(remove_brackets(v)) if type(R)==Series else v
                    ov, oo = o[::-1] 
                    ov = type(v)(ov)

                    if oo=="-":
                        v = v-ov
                    if oo=="+":
                        v = v+ov
                    if can_convert_to_int(v):
                        v = int(v)
                    self.parameter = f"{c} {str(v)}"


        if type(v)==str and c in [">",">=","<","<="]:
            v = convert_to_appropriate_type(remove_brackets(v))
            if type(R) == Series:
                if type(v)!=str and R.dtype=="O":
                    if not all(isinstance(item, list) for item in R if item is not None):
                        R = R.apply(lambda x : convert_to_numeric(x,ls=False))
                    if all(isinstance(item, list) for item in R if item is not None):
                        R = R.apply(convert_to_numeric)

        if type(v)==str and c == "in":
            v = self.split_in(v)
            #if type(R)==Series:

            if type(R.tolist()[0]) == str:
                if R.tolist()[0][-1]==R.tolist()[0][0]=='"':
                    v = [R.dtypes.type(vv) for vv in v]
                else:
                    v = [R.dtypes.type(remove_brackets(vv)) for vv in v]
            else:
                v = [R.dtypes.type(remove_brackets(vv)) for vv in v]

        if type(v)==str and c in ["=","!="] and hasattr(R, "dtypes"):
            v = R.dtypes.type(remove_brackets(v))


        self.R = R
        self.v = v
        self.c = c
        return R, v, c
    
    def split_in(self, text):
        parts = re.split(r",\s*(?=(?:[^']*'[^']*')*[^']*$)", text)
        parts = [part.strip() for part in parts]
        if len(parts)==1:
            parts = text.split(', ')
        parts = [remove_brackets(p) for p in parts]
        return parts

    def EQ(self, R,v):
        if self.is_grouped and self.Noperands==1:
            result =R.apply(lambda x : v in x if x is not None else False)
        else:
            result = R==v
            if type(v) != Series:
                if v == "none" :
                    if hasattr(result, "sum"):
                        if result.sum()==0:
                            result = R.isna()
        return result

    
    def NEQ(self, R,v):
        if self.Noperands==2:
            result = (R!=v) & (~R.isna())
        if self.Noperands==1:
            result = (R!=v) & (~R.isna()) if not self.is_grouped else R.apply(lambda X :  any([x!=v for x in X]))
        return result
    
    def LT(self, R,v):
        if self.Noperands==2:
            try:
                result = R < v
            except:
                result = R < v.tolist()[0]
        if self.Noperands==1:
            result = R < v if not self.is_grouped else R.apply(lambda x : all([(i) < v for i in x]))
        return result
    
    def GT(self, R,v):
        if self.Noperands==2:
            result = R > v
        if self.Noperands==1:
            result = R > v if not self.is_grouped else R.apply(lambda x : all([(i) > v for i in x]))
        return result
    
    def LTE(self, R,v):
        if self.Noperands==2:
            result = R <= v
        if self.Noperands==1:
            result = R <= v if not self.is_grouped else R.apply(lambda x : all([(i) <= v for i in x]))
        return result
    
    def GTE(self, R,v):
        if self.Noperands==2:
            result = R >= v
        if self.Noperands==1:
            result = R >= v if not self.is_grouped else R.apply(lambda x : all([(i) >= v for i in x]))
        return result
    
    def In(self, R, v):
        if not self.is_grouped:
            result = R.isin(v)
        if self.is_grouped:
            result =R.apply(lambda X : any([x in v for x  in X]))
        return result

    def Is(self, R, v):
        if v == "null":
            result = R.isnull()
        if v == "not null":
            result = R.notnull()
        return result
    
    def Intersect(self, R, v):
        if self.is_grouped:
            result = R.apply(lambda X: [x for x in X if x in v])
        else:
            result = R[R.isin(v)]
        return result

    def Union(self, R, v):
        if self.is_grouped:
            result = R.apply(lambda X: list(set(X).union(set(v))))
        else:
            result = pd.concat([R, pd.Series(v)]).drop_duplicates()
        return result

    def Concat(self, R, v):
        result = pd.concat([R, v], axis=1)
        return result

class Aggregation(Node):
    def __init__(self, *args):
        super().__init__(*args)
        self.operands = []  # Ensure operands attri
        
    def execute(self, operand_results=None):
            

        operation_mapping = {
            "max": self.max,
            "min": self.min,
            "distinct": self.distinct,
            "julianday": self.julianday,
            "avg": self.avg,
            "count": self.count,
            "abs": self.abs,
            "sum": self.sum,
            "length" : self.length,
        }

        if self.is_exec == False: 
            self.R = self.convert_to_numeric(operand_results[0])

            if isinstance(self.R, np.int64) or isinstance(self.R, int):
                self.is_grouped = False
            elif hasattr(self.R, '__iter__'):
                self.is_grouped = all(isinstance(item, list) for item in self.R if item is not None)
            else:
                self.is_grouped = False
            

            self.R = operation_mapping[self.parameter](self.R)
            

        self.result = self.R
        return self.result

       
    def max(self, R):
        if self.is_grouped:
            return R.apply(max)
        else:
            return R.max() if isinstance(R,Series) else max(R)
    
    def sum(self, R):
        if self.is_grouped:
            return R.apply(lambda x : np.nansum(x))
        else:
            return R.sum() if isinstance(R,Series) else max(R)
    
    def min(self, R):
        if self.is_grouped:
            return R.apply(min)
        else:
            return R.min() if isinstance(R,Series) else max(R)

    def distinct(self, R):
        if self.is_grouped:
            return R.explode().dropna().unique()
        else:
            return R.dropna().unique()

    def julianday(self, R):
        if self.is_grouped:
            return R.apply(julianday)
        else:
            return R.apply(julianday)
    
    def avg(self, R):

        if self.is_grouped:
            return np.nansum(R)/R.count()
        else:
            return np.nansum(R)/R.count() if isinstance(R,Series) else np.nansum(R)/R.count()
    
    def count(self, R):
        if self.is_grouped:
            if hasattr(R, 'shape') and R.shape[0] == 0:
                return 0
            else:
                return R.apply(len)
        else:
            return R.count() if isinstance(R,Series) else len(R)
    

    def abs(self, R):
        if self.is_grouped:
            raise ValueError(f"Unsupported case: {self.parameter}")
        else:
            return abs(R)
        
    def length(self, R):
        if self.is_grouped:
            raise ValueError(f"Unsupported case: {self.parameter}")
        else:
            return R.str.len()
        
    def convert_to_numeric(self, R):
        self.is_grouped = False if type(R) in [float, int,str, np.float64, np.int64] else all(isinstance(item, list) for item in R)
        #is_groupedgrouped = False if not is_grouped else all(isinstance(item, list) for item in R.iloc[0])

        if self.parameter=="sum":
            if self.is_grouped == False:
                R = R.apply(lambda x : convert_to_numeric(x,ls=False))
        self.R = R
        return R
    




class GroupBy(Node):
    def __init__(self, *args):
        super().__init__(*args)
        self.operands = []  # Specific to ProjectionNode

    def execute(self, operand_results = None):
        try:
            index = [i.idx for i in self.operands].index(True)
        except:
            index = 1 # same nodes

        R1 = operand_results[index]
        R2 = operand_results[1] if index == 0 else operand_results[0]

        is_grouped = all(isinstance(item, list) for item in R1 if item is not None)
        R = pd.DataFrame({'c1': R1, 'c2': R2})
        if is_grouped:
            R['c1_hashable'] = R['c1'].apply(lambda x: tuple(x) if x is not None else None)
            results = R.groupby('c1_hashable', group_keys=True)['c2'].apply(list)
            self.result = results.reset_index()["c2"]
        if not is_grouped:
            self.result = R.groupby('c1')['c2'].apply(list)
        return self.result

        
class Selection(Node):
    def __init__(self, *args):
        super().__init__(*args)
        self.operands = []  # Specific to ProjectionNode

    def execute(self, operand_results=None):

        operation_mapping = {
            "and": self.and_,
            "or": self.or_,
            "where": self.where_,
        }
        if self.is_exec == False: 
            self.extract(operand_results)
            self.R = operation_mapping[self.parameter](self.R, self.B)

        self.result = self.R
        return self.result
    
    def extract(self, operand_results):
        if self.parameter.startswith('and'):
            R, B = operand_results # two bools
            self.parameter = "and"
        if self.parameter.startswith('or'):
            R, B = operand_results # two bools
            self.parameter = "or"
        if self.parameter.startswith('where'):
            if len(operand_results)==2:
                index = operand_results[1].dtypes in [bool, pd.BooleanDtype()]
                R = operand_results[0] if index==1 else operand_results[1]
                B = operand_results[1] if index==1 else operand_results[0]
                self.operands = self.operands if index==1 else self.operands[::-1] # R, B order for linearization
            self.parameter = "where"
        self.R = R
        self.B = B

    def or_(self, R, B ):
        result = R|B
        return result 
    def and_(self, R, B ):
        result = R*B  
        return result 
    def where_(self, R, B ):
        result = R[B].reset_index(drop=True)
        return result 


               
class OrderBy(Node):
    def __init__(self, *args):
        super().__init__(*args)
        self.operands = []  # Specific to ProjectionNode

    def execute(self, operand_results=None):

        if len(operand_results)==2:
            try:
                index = [i.idx for i in self.operands].index(True)
            except:
                index = 1 # same nodes
            R1 = operand_results[index] # col to order with
            R2 = operand_results[1] if index == 0 else operand_results[0]
            R = pd.DataFrame({'R1': R1, 'R2': R2}).dropna()

        direction = self.parameter.split(' ')[1] if self.parameter!="ob" else "ob"
        ascending = False if direction == "desc" else True
        result = R.sort_values(by='R2', ascending=ascending )["R1"] 
        if ascending == False and len(R["R2"].unique())==1:
            all_lists = all(isinstance(item, list) or item is None for item in result)
            if all_lists:
                all_list2 = any(any(isinstance(subitem, list) for subitem in item) for item in result if isinstance(item, list))
                result = result.explode() if all_list2 else result
                result = result.iloc[::-1].apply(lambda x: x[::-1])
            if not all_lists:
                result = result.iloc[::-1]

        self.result = result
        return self.result
        
        
class Limit(Node):
    def __init__(self, *args):
        super().__init__(*args)
        self.operands = []  

    def execute(self, operand_results=None):
        R = operand_results[0]
        K = int(self.parameter.split(" ")[1])
        self.is_grouped = False if type(R) in [float, int,str, np.float64, np.int64] else all(isinstance(item, list) for item in R)
        if self.is_grouped: # flatten group by
            R = R.explode().reset_index(drop=True)
            self.is_grouped = False if type(R) in [float, int,str, np.float64, np.int64] else all(isinstance(item, list) for item in R)
            if self.is_grouped: # flatten list group by
                R = R.explode().reset_index(drop=True)
        if hasattr(R, "iloc"):
            self.result = R.iloc[0:K]
        if type(R) in [int,np.float64,np.int64,float]:
            self.result = R
        self.K = K
        return self.result
    

    
class Operator(Node):
    def __init__(self, *args):
        super().__init__(*args)
        self.operands = []  # Specific to ProjectionNode

    def execute(self, operand_results=None):
        operation_mapping = {
            "+": self.Add,
            "-": self.Sub,
            "/": self.Div
        }
                
        if self.is_exec == False: 
            R,v,o,index = self.extract(operand_results)
            if R is not None:
                result = operation_mapping[o](self.R, self.v, index)
            if R is None:
                result = (o,v)

            self.result = result
        return self.result
        
    def Add(self,R, v, index):
        result = R+v
        return result
    
    def Sub(self, R, v, index):
        if len(index)==1:
            result = R-v if index[0] == 1 else v-R
        if len(index)==2:
            result = R - v
        return result
    
    def Div(self, R, v, index):
        if len(index)==1:
            result = R/v if index[0] == 1 else v/R
        if len(index)==2:
            result = R / v
        return result

    def extract(self, operand_results):
        if len(operand_results)==0: # Operator on future operation
            o = self.parameter.split(' ')[0]
            v = " ".join(self.parameter.split(' ')[1:])
            index = 1
            R = None

        if len(operand_results)==1: # Operator between 1 relations R and one value v
            R = operand_results[0]
            o, index = (self.parameter.split(' ')[0], [1]) if self.parameter.split(' ')[0] in ["-","+"] else (self.parameter.split(' ')[-1], [0])
            v = remove_brackets(self.parameter.replace(o,"").strip())
            v = R.dtypes.type(v) if type(R)==Series else v
            if type(v)==str:
                v = convert_to_appropriate_type(v)

        if len(operand_results)==2: # Operator between on 2 relations R1, R2 (note v)
            try:
                index = [i.idx for i in self.operands].index(True) # check for same node
                index = [i.idx for i in self.operands]
            except:
                index = [0, 1] # same node

            R = operand_results[index.index(True)] # col to order with
            v = operand_results[1] if index.index(True) == 0 else operand_results[0]
            o = self.parameter
            if type(R)==type(v)==str:
                R = convert_to_appropriate_type(R)
                v = convert_to_appropriate_type(v)
            is_grouped = False if type(R) in [float, int,str,np.float64,np.int64] else all(isinstance(item, list) for item in R)
            if not is_grouped  and hasattr(R,"apply") and hasattr(v,"apply"):
                R = R.apply(lambda x : convert_to_numeric(x,ls=False)).reset_index(drop=True) if hasattr(R, 'shape') else R
                v = v.apply(lambda x : convert_to_numeric(x,ls=False)).reset_index(drop=True) if hasattr(v, 'shape') else v
                R = 0 if R.shape[0] == 0 else R
                v = 0 if v.shape[0] == 0 else v

            if is_grouped == True:
                R = R.apply(lambda X : convert_to_numeric(X)).explode()
                v = v.apply(lambda X : convert_to_numeric(X)).explode()

        self.R = R
        self.v = v
        self.o = o
        return R, v, o, index

    

class Having(Node):
    def __init__(self, *args):
        super().__init__(*args)
        self.operands = []  # Specific to ProjectionNode

    def execute(self, operand_results = None):
        G = operand_results[1] if operand_results[1].dtypes != bool else operand_results[0]
        B = operand_results[1] if operand_results[1].dtypes == bool else operand_results[0]
        #self.is_grouped = False if type(G) in [float, int,str,np.float64,np.int64] else all(isinstance(item, list) for item in G)

        try:
            self.result = G[B]
        except:
            self.result = G[B[B].index[0]]

        return self.result



node_classes = {
    'P': Projection,
    'C': Comparison,
    'GB': GroupBy,
    'S': Selection,
    'A': Aggregation,
    'OB': OrderBy,
    'L': Limit,
    "OP" : Operator,
    "H": Having,
}




def operation_order(nodes):
    for name, node in nodes.items():
        operands = node.operands
        if len(operands)==2:
            if operands[0].condition != operands[1].condition:
                operands[0].idx = 1 if operands[0].condition < operands[1].condition else 0
                operands[1].idx = 0 if operands[0].idx==1 else 1
            elif node.prefix == "GB":
                operands[0].idx = 1 if operands[0].position.startswith("group") else 0
                operands[1].idx = 0 if operands[0].idx==1  else 1
            elif node.prefix == "OB":
                operands[0].idx = 0 if operands[0].position.startswith("order") else 1
                operands[1].idx = 1 if operands[0].idx==0  else 0
            elif operands[0].index != operands[1].index:
                operands[0].idx = 0 if operands[0].index ==0 else 1
                operands[1].idx = 1 if operands[0].index ==1  else 0
            else:
                operands[0].idx = 0
                operands[1].idx = 0
            nodes[name].operand = operands

    for name, node in nodes.items():
        operands = node.operands
        for idx in range(len(operands)): # Only node.idx is used for operand position
            operands[idx].condition = None 
            operands[idx].position = None 
            operands[idx].index = None 
        nodes[name].operand = operands
    return nodes


def create_nodes(sql, tbl=None, df=None, mode="wtq"):
    ip = InputProcessor(sql, tbl)
    edges, condi_expressions = create_edges(ip.sql)
    cols = [k.split("|")[-1] for k in list(condi_expressions.keys()) if k.split("|")[0]=="P"]
    if df is None:
        df = ip.get_dataframe(cols)

    nodes = {}
    for tpl in edges:
        for node_str in tpl:
            node_name, prefix, position, index, condition, parameter = node_str.split('|')
            if prefix == "S":
                for s in ["where","and","or"]:
                    parameter = s if parameter.startswith(s) else parameter
            if prefix == "OB":
                parameter = "ob "+parameter
            if prefix == "GB":
                parameter = "gb"
            if prefix == "L":
                parameter = "l "+parameter.split(' ')[1].strip()
            if prefix == "H":
                parameter = "H"
            if prefix == "OP":
                val = condi_expressions["|".join(node_str.split('|')[1:])]
                if type(val)==tuple:
                    parameter = translate(val[0])+" "+translate(val[1])
            nodes[node_name] = node_classes.get(prefix)(node_name, parameter, prefix, df, position, condition, index, mode)
    
    if len(condi_expressions)==1:
        single_node=list(condi_expressions.keys())[0]
        node_name, prefix, position, index, condition, parameter = ["N1"]+single_node.split('|')
        nodes[node_name] = node_classes.get(prefix)(node_name, parameter, prefix, df, position, condition, index, mode)

    edges_new = []
    for n1, n2 in edges:
        n1 = n1.split('|')[0]
        n2 = n2.split('|')[0]
        edges_new.append((n1,n2))
        nodes[n2].operands.append(nodes[n1])
        
    nodes = operation_order(nodes)

    G = Graph(edges_new, nodes, ip.sql)
    if df is not None:
        G.header = df.columns.tolist()

    return G



