

import re 
import pandas as pd
import sqlite3





class InputProcessor:
    def __init__(self, sql, tbl=None):
         
        self.sql, check = self.get_sql_query(sql)
        self.check = check
        self.tbl = tbl

    def get_table_dict(self,tbl):
        if tbl is not None:
            db_file = f"/home/raphael.gervillie/sql_graph/data/tables/db/{tbl}.db"
            conn = sqlite3.connect(db_file)
            df = pd.read_sql_query("SELECT * FROM w", conn)
            return df
        else:
            return None
    
    def get_dataframe(self, cols):
        if self.tbl is not None:
            df = self.get_table_dict(self.tbl)
            if self.check:
                df = self.replace_values_in_table(df)
            df = prep_df(df, cols)
            self.df = df
            return df
        else:
            return None

    def get_sql_query(self, sql):
        sql = sql.replace('not null',"is not null")
        sql = sql.replace('present_ref',"2014")
        sql = sql.replace(' * ', ' id ')   
        
        check = False

        r = ('\\\'',' ')
        if r[0] in sql:
            check=True
            
            sql = sql.replace(r[0], r[1])

        return sql, check

    def replace_values_in_table(self, table):
        for column in table.columns:
            if table[column].dtype == 'object':
                table[column] = table[column].apply(lambda x: str(x).replace("\'"," "))
        return table  
    



def find_split_pattern(df, c):
    N1 = df[c].apply(lambda x : x.split(',') if x is not None else x).explode().shape
    N2 = df[c].apply(lambda x : x.split('-') if x is not None else x).explode().shape
    N3 = df[c].apply(lambda x : x.split('\n') if x is not None else x).explode().shape
    if N1>=N2 and N1>=N3:
        return ","
    if N2>=N1 and N2>=N3:
        return "-"
    if N3>=N1 and N3>=N2:
        return "\n"
    
def split_to_list(value, splitter):
    if pd.notna(value):
        return [i.strip() for i in value.split(splitter)]
    return [value]


def list_first(row):
    matches = re.findall(r'([\w\s°]+)(?:\s*\([^)]*\))?', row)
    return [match.strip() for match in matches if match]


def extract_years(date):
    # Use regular expression to find the year part of the date
    match = re.search(r'\b\d{4}\b', str(date))
    if match:
        return match.group()
    else:
        return None
    
def extract_month(date):
    # Use regular expression to find the month part of the date
    match = re.search(r'([a-zA-Z]+)', str(date))
    if match:
        return month_to_number.get(match.group().lower())  # Convert to lowercase and use the mapping
    else:
        return None
    

month_to_number = {
    'january': '1',
    'february': '2',
    'march': '3',
    'april': '4',
    'may': '5',
    'june': '6',
    'july': '7',
    'august': '8',
    'september': '9',
    'october': '10',
    'november': '11',
    'december': '12'
}

def prep_df(df, cols):
    
    check_address = sum(["_address" in c for c in cols])>0
    if check_address:
        indices = [index for index, item in enumerate(cols) if "_address" in item]
        col_address = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_address:
             df[c] = df[c.split('_address')[0]].str.split(', ')

    check_list = sum(["_list" in c and "_list_" not in c  for c in cols])>0    
    if check_list:
        indices = [index for index, item in enumerate(cols) if "_list" in item and "_list_" not in item ]
        col_list = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_list:
            split_pattern = find_split_pattern(df, c.split('_list')[0])
            df[c] = pd.Series([[subitem for item in sublist if item is not None for subitem in item.split('/') if subitem] for sublist in df[c.split('_list')[0]].apply(lambda x : split_to_list(x,split_pattern))])

    check_list = sum(["_list_maximum_number" in c for c in cols])>0    
    if check_list:
        indices = [index for index, item in enumerate(cols) if "_list_maximum_number" in item]
        col_list = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_list:
            split_pattern = find_split_pattern(df, c.split('_list')[0])
            df[c] = df[c.split('_list_maximum_number')[0]].apply(lambda x : max(split_to_list(x, split_pattern)))
    check_list = sum(["_list_minimum_number" in c for c in cols])>0    
    if check_list:
        indices = [index for index, item in enumerate(cols) if "_list_minimum_number" in item]
        col_list = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_list:
            split_pattern = find_split_pattern(df, c.split('_list')[0])
            df[c] = df[c.split('_list_minimum_number')[0]].apply(lambda x : min(split_to_list(x, split_pattern)))
            
    check_list = sum(["_list_number" in c for c in cols])>0    
    if check_list:
        indices = [index for index, item in enumerate(cols) if "_list_number" in item]
        col_list = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_list:
            df[c] = df[c.split('_list_number')[0]].str.extract(r'(\d+\.\d+|\d+)', expand=False).astype(float)
            
    check_list = sum(["_list_maximum_year" in c for c in cols])>0    
    if check_list:
        indices = [index for index, item in enumerate(cols) if "_list_maximum_year" in item]
        col_list = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_list:
            df[c] = df[c.split('_list_maximum_year')[0]].apply(extract_max_year)
  
    check_list = sum(["_list_minimum_year" in c for c in cols])>0    
    if check_list:
        indices = [index for index, item in enumerate(cols) if "_list_minimum_year" in item]
        col_list = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_list:
            df[c] = df[c.split('_list_minimum_year')[0]].apply(extract_min_year)

    check_list = sum(["_list_minimum" in c and "_list_minimum_" not in c  for c in cols])>0    
    if check_list:
        indices = [index for index, item in enumerate(cols) if "_list_minimum" in item]
        col_list = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_list:
            df[c] = df[c.split('_list_minimum')[0]].apply(extract_minimum)
  
    check_list = sum(["_list_maximum" in c and "_list_maximum_" not in c  for c in cols])>0    
    if check_list:
        indices = [index for index, item in enumerate(cols) if "_list_maximum" in item]
        col_list = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_list:
            df[c] = df[c.split('_list_maximum')[0]].apply(extract_maximum)

    check_list = sum(["_list_first" in c for c in cols])>0    
    if check_list:
        indices = [index for index, item in enumerate(cols) if "_list_first" in item]
        col_list = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_list:
            df[c] = df[c.split('_list_first')[0]].apply(list_first)
            

    check_list = sum(["_list_year" in c for c in cols])>0    
    if check_list:
        indices = [index for index, item in enumerate(cols) if "_list_year" in item]
        col_list = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_list:
            df[c] = df[c.split('_list_year')[0]].apply(extract_years)

    check_list = sum(["_list_month" in c for c in cols])>0    
    if check_list:
        indices = [index for index, item in enumerate(cols) if "_list_month" in item]
        col_list = [c for idx,c in enumerate(cols) if idx in indices]
        for c in col_list:
            df[c] = df[c.split('_list_month')[0]].apply(extract_month)

    return df

def extract_max_year(row):
    years = row.split(', ') if row is not None else None
    max_years = []
    if years is not None:
        for year_range in years:
            year_range = year_range.split('-')
            if len(year_range) == 1:
                max_years.append(extract_years(year_range[0]))
            else:
                max_years.append(str(max(map(int, [extract_years(y) for y in year_range]))))
    else:
        max_years.append(None)
    return max_years

def extract_min_year(row):
    years = row.split(', ') if row is not None else None
    min_years = []
    if years is not None:
        for year_range in years:
            year_range = year_range.split('-')
            if len(year_range) == 1:
                min_years.append(extract_years(year_range[0]))
            else:
                min_years.append(str(min(map(int, [extract_years(y) for y in year_range]))))
    else:
        min_years.append(None)

    return min_years


def extract_minimum(row):
    years = row.split(', ') if row is not None else None
    min_years = []
    if years is not None:
        for year_range in years:
            year_range = year_range.split('-')
            if len(year_range) == 1:
                min_years.append(year_range[0])
            else:
                min_years.append(str(min(map(int, year_range))))
    else:
        min_years.append(None)
    return min_years

def extract_maximum(row):
    years = row.split(', ') if row is not None else None
    max_years = []
    if years is not None:
        for year_range in years:
            year_range = year_range.split('-')
            if len(year_range) == 1:
                max_years.append(year_range[0])
            else:
                max_years.append(str(max(map(int, year_range))))
    else:
        max_years.append(None)
    return max_years




