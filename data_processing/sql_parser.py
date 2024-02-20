
from sqlglot import exp, parse_one, select

from .utils import   translate, can_convert_to_int


def create_expression_key(prefix, parameter, position, index, condition):
    return f"{prefix}|{position}|{index}|{condition}|{parameter}"

def remove_aggregator(expression):
    aggregators = [exp.Count,exp.Min, exp.Max, exp.Sum, exp.Avg, exp.Length]
    special_tokens = []
    for aggregator in aggregators:
        if isinstance(expression, aggregator):
            expression = next(iter(expression.find_all(aggregator))).this
            special_tokens.append(aggregator)
            
        if isinstance(expression, exp.Anonymous):
            expression = next(iter((next(iter(expression.find_all(exp.Anonymous))).find_all(exp.Identifier))))
            special_tokens.append(exp.Anonymous)
    return expression, special_tokens

def duplicate(expressions, expression, position, condition, name):
    for position, e in expression:
        parameter = translate(e)
        expressions[f'{name},{parameter},{position},{condition}'] = e
    return expressions

def where_expression(expression):
    selections = {exp.GT, exp.LT, exp.GTE, exp.LTE, exp.EQ, exp.NEQ}
    projection = None  
    selection = None  
    abs_ = None
    for s in selections:
        if isinstance(expression, s):
            if isinstance(expression.this, exp.Abs):
                abs_ = "abs"
            projection = expression.this
            selection = (expression.expression, s)

    if isinstance(expression, exp.In) :
        projection = expression.this
        selection = (expression.expressions, exp.In)

    if isinstance(expression, exp.Is):
        projection = expression.this
        selection = ("null", exp.Is)
        
    if isinstance(expression, exp.Not):
        projection = expression.this.this
        selection = ("not null", exp.Is)
    return projection, selection, abs_



def where_expressions(expression):
    projections = []
    selections = []    
    abss = []
    X = []

    def traverse_expression(expression):
        nonlocal projections, selections, abss, X
        if isinstance(expression, exp.Paren):
            expression = expression.this
        

        if isinstance(expression, exp.And):
            X.append("AND")  # Append "AND" before traversing sub-expressions
            for k, expr in expression.iter_expressions():
                traverse_expression(expr)
        elif isinstance(expression, exp.Or):
            X.append("OR")   # Append "OR" before traversing sub-expressions
            for k, expr in expression.iter_expressions():
                traverse_expression(expr)
        else:
            projection, selection, abs_ = where_expression(expression)
            selections.append(selection)
            projections.append(projection)
            abss.append(abs_)
     
    traverse_expression(expression)
        
    return projections, selections, abss, X


def having_expression(expression):
    expression = expression.this
    selections = {exp.GT, exp.LT, exp.GTE, exp.LTE, exp.EQ}
    projection = None  # Default value
    selection = None   # Default value
    for s in selections:
        if isinstance(expression, s):
            projection = expression.this
            selection = (expression.expression,s)
    return projection, selection
    
def where_check_subquery(selection):
    selection, operator = where_check_operator(selection)
    check=False
    subquery = None
    if isinstance(selection[0], exp.Subquery):
        check = True
        subquery, selection = selection
        subquery = subquery.this
    return check, selection, subquery, operator

def where_check_operator(selection):
    operators = {exp.Sub, exp.Add}
    operator = None
    operator_ls=[]
    selection2 = selection
    for o in operators:
        if isinstance(selection[0], o):
            operator = (o, selection[0].expression)
            s2 = selection[0].this
            selection2 = (s2, selection[1])
    if operator is not None:
        operator_ls.append(operator)
    return selection2, operator_ls

def is_quoted_(selection):
    is_quoted=False
    if selection.sql()[0]==selection.sql()[-1]=='"':
        is_quoted=True
    if selection.sql()[0]==selection.sql()[-1]=="'":
        is_quoted=True
    return is_quoted

def parse_expression(expressions, expression, all_keys, key, semi, condition, idid=0):

    if key=="expressions":
        position = f"select{semi}"
        
        expressions, expression = extract_abs(expressions, expression, position, condition)
        expression, aggr = remove_aggregator(expression)
        if aggr is not None:
            expressions = add_expressions(expressions, aggr, "A", position+"*", condition)
        expressions, expression = extract_abs(expressions, expression, position+"1", condition)


        expressions, expression = multi_expressions(expressions, expression, position, condition, 0)
        if f"where{semi}" in all_keys:
            expressions = add_expressions(expressions, [f"where{semi}p{idid}" for z in range(len(expression))], "S", position, condition)
        
        expressions, expression = remove_aggregators(expressions, expression, position, condition)
        expressions, expression = extract_distinct(expressions, expression, position, condition) 
        expressions = add_expressions(expressions, expression, "P", position, condition)
        return expressions
    
    

    if key=="order":
        position = f"order{semi}"
        #expressions = add_expressions(expressions, ["OB"], "OB", position, condition)
        
        expressions, expression, direction_ls = directions(expressions, expression, position, condition)       

        expressions = add_expressions(expressions, direction_ls, "OB", position, condition)
        if len(direction_ls)==0:
            expressions = add_expressions(expressions, ["OB"], "OB", position, condition)
            

        expressions, expression = extract_abs(expressions, expression, position, condition)
        expression, aggr = remove_aggregator(expression)
        if aggr is not None:
            expressions = add_expressions(expressions, aggr, "A", position+"*", condition)
        expressions, expression = extract_abs(expressions, expression, position+"1", condition)

        
        expressions, expression = multi_expressions(expressions, expression, position, condition, 0)
        
        if f"where{semi}" in all_keys:
            expressions = add_expressions(expressions, [f"where{semi}o" for z in range(len(expression))], "S", position, condition)
        if f"group{semi}" in all_keys:
            expressions = add_expressions(expressions, [f"group{semi}gb" for z in range(len(expression))], "GB", position, condition)    
           
        expressions, expression = remove_aggregators(expressions, expression, position, condition)
        expressions, expression = extract_distinct(expressions, expression, position, condition)        
        expressions = add_expressions(expressions, expression, "P", position, condition)
        return expressions

    
    if key=="where":
        position = f"where{semi}"
        projections, selections, abss, X = where_expressions(expression)
        for z0, (projection, selection, a) in enumerate(zip(projections, selections, abss)): # Iterate over AND

            if z0==1:
                and_or = X[z0-1] if len(X)<3 else X[z0]
                expressions = add_expressions(expressions, [f"{and_or}{z0}"], "S", position, condition)
            if z0==2 and len(X)<3:
                and_or = X[z0-1] if len(X)<3 else X[z0]
                expressions = add_expressions(expressions, [f"{and_or}{z0}"], "S", position, condition)
                expressions = add_expressions(expressions, [f"{and_or}*"], "S", position+"agg", condition)
            if z0==3 and len(X)==3:
                expressions = add_expressions(expressions, [f"{X[z0-1]}{z0-1}"], "S", position, condition)
                expressions = add_expressions(expressions, [f"{X[0]}*"], "S", position+"agg", condition)
            
            expressions, projection = extract_abs(expressions, projection, position, condition)
            expressions, projection = multi_expressions(expressions, projection, position, condition, z0)
            expressions, projection = remove_aggregators(expressions, projection, position, condition)
            expressions = add_expressions(expressions, projection, "P", position+str(z0), condition)
            check, selection, subquery, operator = where_check_subquery(selection)
            if type(selection)==tuple:
                if isinstance(selection[0],exp.Column) and not is_quoted_(selection[0]) :
                    k = create_expression_key("C", translate(selection[1]), position, str(z0), condition)
                    expressions[k] = selection[1]
                    k = create_expression_key("P", translate(selection[0]), position+"c"+str(z0), str(z0), condition)
                    expressions[k] = selection[0]
                    
                else:
                    k = create_expression_key("C", translate(selection), position, str(z0), condition)
                    expressions[k] = selection
            else:
                k = create_expression_key("C", translate(selection), position, str(z0), condition)
                expressions[k] = selection

            expressions = add_expressions(expressions, operator, "OP", position, condition)

            if check:
                all_semi_keys = [k+"semi" for k,v in subquery.iter_expressions()]
                for semi_key, semi_expression  in subquery.iter_expressions(): # For semi-query
                    semi_expression = extract(semi_key, semi_expression)
                    expressions = parse_expression(expressions, semi_expression, all_semi_keys, semi_key, "semi", condition, z0)
        return expressions

    if key=="limit":
        position = f"limit{semi}"
        expressions = add_expressions(expressions, [expression], "L", position, condition)
        return expressions
    
    if key=="distinct":
        position = f"select{semi}"
        expressions = add_expressions(expressions, [expression], "A", position, condition)
        return expressions
    
    if key=="group":
        position = f"group{semi}"
        expressions = add_expressions(expressions, ["GB"], "GB", position, condition)
        expression = expression.expressions[0].this
        expressions = add_expressions(expressions, [expression], "P", position, condition)     

        if f"where{semi}" in all_keys:
            expressions = add_expressions(expressions, [f"where{semi}gb"], "S", position, condition)

        return expressions
    
    if key == "having":
        position = f"having{semi}"
        
        expressions = add_expressions(expressions, ["having"], "H", position, condition)
        projections, selections = having_expressions(expression)
        for z0, (projection, selection) in enumerate(zip(projections, selections)): # Iterate over AND
            #expressions = add_expressions(expressions, ["GB"], "GB", position, condition)
            if z0>=1:
                expressions = add_expressions(expressions, [f"AND{z0}"], "H", position+f"AND{z0}", condition)
                

            expressions, projection = multi_expressions(expressions, projection, position, condition, z0)
            if f"where{semi}" in all_keys:
                expressions = add_expressions(expressions, [f"where{semi}o" for z in range(len(projection))], "S", position, condition)
            if f"group{semi}" in all_keys:
                expressions = add_expressions(expressions, [f"group{semi}gb" for z in range(len(projection))], "GB", position, condition)    

        
            expressions, projection = remove_aggregators(expressions, projection, position, condition, z0)
            expressions = add_expressions(expressions, projection, "P", position+str(z0), condition)
            check, selection, subquery, operator = where_check_subquery(selection)            
            k = create_expression_key("C", translate(selection), position, str(z0), condition)
            expressions[k] = selection
            expressions = add_expressions(expressions, operator, "OP", position, condition)
    
            if check:
                all_semi_keys = [k+"semi" for k,v in subquery.iter_expressions()]
                for semi_key, semi_expression  in subquery.iter_expressions(): # For semi-query
                    semi_expression = extract(semi_key, semi_expression)
                    expressions = parse_expression(expressions, semi_expression, all_semi_keys, semi_key, "semi", condition, idid)
                    
        
        return expressions
        
    else:
        return expressions
    

def extract(key, expression):
    if key =="where":
        expression = next(iter(expression.find_all(exp.Where))).this
    if key =="order":
        expression = next(iter(expression.find_all(exp.Order))).expressions[0]
    if key =="expressions":
        if isinstance(expression, exp.Paren):

            expression = expression.this
    return expression


def having_expressions(expression):
    projections = []
    selections = []    
    if isinstance(expression.this, exp.And):
        for k,expr in expression.this.iter_expressions():
            projection, selection = having_expression(expr)
            selections.append(selection)
            projections.append(projection)

    if isinstance(expression.this, exp.Or):
        for k,expr in expression.this.iter_expressions():
            projection, selection = having_expression(expr)
            selections.append(selection)
            projections.append(projection)
            
    if not isinstance(expression.this, exp.And) and not isinstance(expression.this, exp.Or):
        projection, selection = having_expression(expression.this)
        selections.append(selection)
        projections.append(projection)

    return projections, selections

def having_expression(expression):
    selections = {exp.GT, exp.LT, exp.GTE, exp.LTE, exp.EQ, exp.NEQ}
    projection = None  # Default value
    selection = None   # Default value
    for s in selections:
        if isinstance(expression, s):
            projection = expression.this
            selection = (expression.expression,s)
    return projection, selection


def directions(expressions, expression, position, condition):
    direction_ls = []
    direction = None
    string = str(expression)
    if string[-3:]=="ASC":
        direction = "asc"
    if string[-4:]=="DESC":
        direction = "desc"
    if direction is not None:
        direction_ls.append(direction)

    return expressions, expression.this, direction_ls

def extract_distinct(expressions, expression, position, condition):
    distinct_ls=[]
    for idx, e in enumerate(expression):
        distinct = None
        if isinstance(e, exp.Distinct):
            e = e.expressions[0].this
            distinct = exp.Distinct
            expression[idx] = e
        if distinct is not None:
            distinct_ls.append(distinct)    
    expressions = add_expressions(expressions, distinct_ls, "A", position, condition)
    return expressions, expression

def remove_aggregators(expressions,expression, position, condition, z0=None):
    expressions_rmv = []
    aggregation = []
    for express in expression:
        e, a = remove_aggregator(express)
        if e is not None:
            expressions_rmv.append(e)
        aggregation.extend(a)
    if z0 is not None:
        expressions = add_expressions(expressions, aggregation, "A", position+str(z0), condition)
    if z0 is None:
        expressions = add_expressions(expressions, aggregation, "A", position, condition)
    return expressions, expressions_rmv

def multi_expressions(expressions, parsing, position, condition, idx):
    global_comparators = [exp.GT, exp.LT, exp.GTE, exp.LTE, exp.Sub, exp.Add, exp.Div]
    operator = []
    sub_query = []
    for global_comparator in global_comparators:
        if isinstance(parsing, global_comparator):
            expression = next(iter(parsing.find_all(global_comparator)))
            
            e1 = expression.this
            e2 = expression.expression
            checke1 = [isinstance(e1, global_comparator) for global_comparator in global_comparators ]
            checke2 = [isinstance(e2, global_comparator) for global_comparator in global_comparators ]
            if sum(checke1)>0:
                index = checke1.index(True)
                global_comparator2 = global_comparators[index]
                e3 = e1.expression
                e1 = e1.this
                operator.append(global_comparator2)
                sub_query.append(e1)
                sub_query.append(e3)
                sub_query.append(e2)
                operator.append(global_comparator)
                
            elif sum(checke2)>0:
                index = checke2.index(True)
                global_comparator2 = global_comparators[index]
                e3 = e2.expression
                e2 = e2.this
                sub_query.append(e3)
                operator.append(global_comparator2)
                sub_query.append(e2)
                sub_query.append(e1)
                operator.append(global_comparator)
                
            else:
                sub_query.append(e1)
                sub_query.append(e2)
                operator.append(global_comparator)

                
    if len(sub_query)>1:
        check = any([type(s) == exp.Literal for s in sub_query])
        if not check: # two cols or more
            for id_, oo in enumerate(operator):
                
                expressions = add_expressions(expressions, [oo], "OP", position+str(id_), condition, idx)
            return expressions, sub_query
        if check: # There is col alteration ex : c3 - 2008
            index = [type(s) == exp.Literal for s in sub_query].index(True)
            operator  = (sub_query[index], operator[0]) if index==0 else (operator[0], sub_query[index])
            sub_query = [sub_query[[type(s) == exp.Literal for s in sub_query].index(False)]]

            if  translate(operator[0]) in [">",">=","<","<=","=","!="]:
                expressions = add_expressions(expressions, [operator], "C", position, condition, idx)
            else : 
                expressions = add_expressions(expressions, [operator], "OP", position, condition, idx)
            return expressions, sub_query
    else:
        return expressions, [parsing]
    
    
def extract_abs(expressions, expression, position, condition):
    abs_expression = []
    if isinstance(expression, exp.Abs):
        expression=next(iter((expression.find_all(exp.Abs)))).this
        abs_expression = [exp.Abs]
        
    expressions = add_expressions(expressions, abs_expression, "A", position+"abs", condition)
    return expressions, expression


def create_expression_key(prefix, parameter, position, index, condition):
    if prefix == "S":
        return f"{prefix}|{position}{parameter}|{index}|{condition}|{parameter}"
    
    if prefix == "A" and parameter in ["abs","distinct"]:
        return f"{prefix}|{position}{parameter}|{index}|{condition}|{parameter}"
        
    #print(f"{prefix},{parameter},{position},{index},{condition}")
    return f"{prefix}|{position}|{index}|{condition}|{parameter}"

def add_expressions(expressions, items, prefix, position, condition, idx=None):
    if items is not None:
        for i, item in enumerate(items):
            parameter = translate(item)
            if idx is not None:
                key = create_expression_key(prefix, parameter, position, idx, condition)
            else : 
                key = create_expression_key(prefix, parameter, position, i, condition)
            expressions[key] = item
    return expressions




def global_operator(parsing):
    global_comparator=None
    if isinstance(parsing, exp.Expression):
        p = parsing.expressions[0]
        global_comparators = [exp.GT, exp.LT, exp.GTE, exp.LTE, exp.Sub, exp.Add]
        for gc in global_comparators:
            if isinstance(p, gc):
                last_digit = p.expression
                global_comparator = (gc, last_digit)
    parsing = p.this         
    return parsing, global_comparator




def digit_at_end_(parsing, all_=False):
    global_co = [exp.GT, exp.LT, exp.GTE, exp.LTE, exp.EQ, exp.NEQ] + [exp.Sub, exp.Add]

    digit_at_end = None
    digit_at_start = None
    global_abs = False

    is_digit=False
    is_comparator=False
    if isinstance(parsing, exp.Select) and len([k for k in parsing.find_all(exp.Select)])>1 :
        p1 = parsing.expressions[0]
        global_abs = isinstance(p1,exp.Abs)
        for gg in global_co:
            if isinstance(p1, gg):
                if can_convert_to_int(p1.expression.sql()) or isinstance(p1.expression, exp.Literal):
                    digit_at_end = (gg, p1.expression)
                    global_abs = isinstance(p1.this,exp.Abs)
                    is_digit=True
                    return is_digit,is_comparator, digit_at_end, digit_at_start, global_abs
                
                if can_convert_to_int(p1.this.sql()) or isinstance(p1.this, exp.Literal):
                    digit_at_start = (p1.this, gg)
                    global_abs = isinstance(p1.this,exp.Abs)
                    return is_digit,is_comparator, digit_at_end, digit_at_start, global_abs
                
                if can_convert_to_int(p1.expression.sql()) ==False and isinstance(p1.expression, exp.Literal)==False :
                    is_digit=False
                    return is_digit,is_comparator, digit_at_end, digit_at_start, global_abs
                
        if isinstance(p1, exp.Is):
            is_comparator=True
            digit_at_end = ("null", exp.Is)
            return is_digit,is_comparator, digit_at_end, digit_at_start, global_abs

        if isinstance(p1, exp.Not):
            is_comparator=True
            digit_at_end = ("not null", exp.Is)
            return is_digit,is_comparator, digit_at_end, digit_at_start, global_abs
        
    if all_==True:
        p1 = parsing
        for gg in global_co:
            if isinstance(p1, gg):
                if can_convert_to_int(p1.expression.sql()):
                    digit_at_end = (gg, p1.expression)
                    global_abs = isinstance(p1.this,exp.Abs)
                    is_digit=True
                    return is_digit,is_comparator, digit_at_end, digit_at_start, global_abs
                if can_convert_to_int(p1.expression.sql()) ==False:
                    is_digit=False
                    return is_digit,is_comparator, digit_at_end, digit_at_start, global_abs
                
    return is_digit,is_comparator, digit_at_end, digit_at_start, global_abs

def parsing_subquerys_global(query):
    expressions_global = {}
    global_comparators = [exp.GT, exp.LT, exp.GTE, exp.LTE, exp.EQ, exp.NEQ]
    global_operators = [exp.Sub, exp.Add]
    check2 = query.startswith('select count ( * ) from ( select')
    sub_query = []
    parsing = parse_one(query)
    Nsubquerys = len(list((parsing.find_all(exp.Subquery))))
    is_digit, is_comparator, digit_at_end, digit_at_start, global_abs = digit_at_end_(parsing)

    if is_digit:

        if  translate(digit_at_end[0]) in [">",">=","<","<=","=","!="]:
            expressions_global = add_expressions(expressions_global, [digit_at_end], "C", "global_end", 0, 0)
        else:
            expressions_global = add_expressions(expressions_global, [digit_at_end], "OP", "global_end", 0, 0)

    if is_comparator:
        expressions_global = add_expressions(expressions_global, [digit_at_end], "C", "global_end", 0, 0)

    if digit_at_start is not None:
        if  translate(digit_at_start[1]) in [">",">=","<","<="]:
            if translate(digit_at_start[1]) == "<":
                digit_at_start2 = (digit_at_start[0],exp.GT)
            if translate(digit_at_start[1]) == ">":
                digit_at_start2 = (digit_at_start[0],exp.LT)
            if translate(digit_at_start[1]) == "<=":
                digit_at_start2 = (digit_at_start[0],exp.GTE)
            if translate(digit_at_start[1]) == ">=":
                digit_at_start2 = (digit_at_start[0],exp.LTE)

            expressions_global = add_expressions(expressions_global, [digit_at_start2], "C", "global_end", 0, 0)
        else : 
            expressions_global = add_expressions(expressions_global, [digit_at_start], "OP", "global_end", 0, 0)


    if not check2 and Nsubquerys >=2:
        for global_ in global_operators+global_comparators:
            for glob in parsing.find_all(global_):
                if len(list((glob.find_all(exp.Subquery))))==2:
                    is_digit, _, _, _, _ = digit_at_end_(glob,all_=True)

                    if not is_digit:
                        for select in glob.find_all(exp.Subquery):

                            sub_query.append(select.this)
                            if global_ in global_operators:
                                expressions_global = add_expressions(expressions_global, [global_], "OP", "global", 0, 0)

                            if global_ in global_comparators:
                                expressions_global = add_expressions(expressions_global, [global_], "C", "global", 0, 0)

    if " intersect " in query:
        for select in parsing.find_all(exp.Select):
            sub_query.append(select)
            expressions_global = add_expressions(expressions_global, [exp.Intersect], "C", "global", 0, 0)

    if " union " in query and isinstance(parsing, exp.Union):
        for select in parsing.find_all(exp.Select):
            sub_query.append(select)
            expressions_global = add_expressions(expressions_global, [exp.Union], "C", "global", 0, 0)




                
    
    if check2:
        sub_query.append(next(iter(parsing.find_all(exp.From))).this.this)
        expressions_global = add_expressions(expressions_global, [exp.Count], "A", "global", 0, 0)
        return sub_query, expressions_global
    
    if len(sub_query)>1:
        if global_abs:
            expressions_global = add_expressions(expressions_global, [exp.Abs], "A", "global", 0, 0)
        return sub_query, expressions_global
    #if len(sub_query)==0 and digit_at_end is not None:
    #    return [parsing.expressions[0].this.this],expressions_global
    
    if len(sub_query)==0 and digit_at_end is not None:
        parsing = parsing.expressions[0].this
        parsing = parsing.this if isinstance(parsing, exp.Subquery) else select(parsing)
        return  [parsing], expressions_global

    if len(sub_query)==0 and digit_at_start is not None:
        parsing = parsing.expressions[0].expression
        parsing = parsing.this if isinstance(parsing, exp.Subquery) else select(parsing)
        return  [parsing], expressions_global  
    
    else:
        return [parsing], {}
    



def modif_expression(expressions, all_keys):
    if sum(["|".join(a.split('|')[:4]) == "P|select|0|0" for a in list(expressions.keys())])>1: # c1, c2
        expressions = add_expressions(expressions, [","], "C", "select", 0, 0)
        if "order" in all_keys:
            index = [i.startswith('OB|order|0') for i in list(expressions.keys())].index(True)
            name = list(expressions.keys())[index]
            bool_ls = [i.startswith('P|select|0') for i in list(expressions.keys())]
            last_true = max(loc for loc, val in enumerate(bool_ls) if val) if any(bool_ls) else None

            name2 = list(expressions.keys())[last_true] # modif the name
            val2 = expressions[name2]
            expressions.pop(name2)
            name2 = "|".join(name2.split('|')[:2])+"|1|"+"|".join(name2.split('|')[3:])
            expressions[name2] = val2

            expressions['OB|order|1|'+"|".join(name.split("|")[3:])] = expressions[name]  

        if "where" in all_keys: 
            index = [i.startswith('S|selectwherep0|0') for i in list(expressions.keys())].index(True)
            name = list(expressions.keys())[index]
            bool_ls = [i.startswith('P|select|0') for i in list(expressions.keys())]
            last_true = max(loc for loc, val in enumerate(bool_ls) if val) if any(bool_ls) else None

            name2 = list(expressions.keys())[last_true] # modif the name
            val2 = expressions[name2]
            expressions.pop(name2)
            name2 = "|".join(name2.split('|')[:2])+"|1|"+"|".join(name2.split('|')[3:])
            expressions[name2] = val2

            expressions['S|selectwherep0|1|'+"|".join(name.split("|")[3:])] = expressions[name]  

    return expressions


def parse_query(query):
    sub_query, expressions_global = parsing_subquerys_global(query)
    expressions={}
    for condition, parsing in enumerate(sub_query):       
        all_keys = [k for k,v in parsing.iter_expressions()]
        for key, expression  in parsing.iter_expressions(): 
            expression = extract(key, expression)
            expressions = parse_expression(expressions, expression, all_keys, key, semi="", condition=str(condition))
    expressions = modif_expression(expressions, all_keys)
    return expressions, expressions_global
