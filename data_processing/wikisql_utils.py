


# coding=utf-8
# Copyright 2022 The Microsoft, The Google and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import enum
import functools
import math
import re
import pandas as pd
from copy import deepcopy


# The following script is adapted from the script of TaPas.
# Original: https://github.com/google-research/tapas/master/wikisql_utils.py
from typing import Any, List, Text


EMPTY_ANSWER = "none"
EMPTY_ANSWER_AGG = "none"

def transform_condition(input_dict):

    column_indexes = input_dict.get('column_index', [])
    operator_indexes = input_dict.get('operator_index', [])
    conditions = input_dict.get('condition', [])    
    transformed_output = [[col_idx, op_idx, cond] for col_idx, op_idx, cond in zip(column_indexes, operator_indexes, conditions)]

    return transformed_output




def change_col_type(df, col_types):
    type_mapping = {
        'text': 'object',  # or 'string' for pandas >= 1.0.0
        'real': 'float'
    }
    for col, col_type in zip(df.columns, col_types):
        if col_type=="real":
            df[col] = df[col].astype(type_mapping[col_type])            
        if col_type=="text":
            df[col] = df[col].str.lower()      
    return df



def _split_thousands(delimiter, value):
    split = value.split(delimiter)
    return len(split) > 1 and any((len(x) == 3 for x in split))


def convert_to_float(value):
    """Converts value to a float using a series of increasingly complex heuristics.
    Args:
      value: object that needs to be converted. Allowed types include
        float/int/strings.
    Returns:
      A float interpretation of value.
    Raises:
      ValueError if the float conversion of value fails.
    """
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return float(value)
    if not isinstance(value, str):
        raise ValueError("Argument value is not a string. Can't parse it as float")
    sanitized = value

    try:
        # Example: 1,000.7
        if "." in sanitized and "," in sanitized:
            return float(sanitized.replace(",", ""))
        # 1,000
        if "," in sanitized and _split_thousands(",", sanitized):
            return float(sanitized.replace(",", ""))
        # 5,5556
        if "," in sanitized and sanitized.count(",") == 1 and not _split_thousands(",", sanitized):
            return float(sanitized.replace(",", "."))
        # 0.0.0.1
        if sanitized.count(".") > 1:
            return float(sanitized.replace(".", ""))
        # 0,0,0,1
        if sanitized.count(",") > 1:
            return float(sanitized.replace(",", ""))
        return float(sanitized)
    except ValueError:
        # Avoid adding the sanitized value in the error message.
        raise ValueError("Unable to convert value to float")


def _normalize_float(answer):
    if answer is None:
        return None
    try:
        value = convert_to_float(answer)
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    except ValueError:
        return answer.lower()


_TYPE_CONVERTER = {
    "text": lambda x: x,
    "real": convert_to_float,
}


class _Aggregation(enum.Enum):
    """Aggregations as defined by WikiSQL. Indexes match the data."""

    NONE = 0
    MAX = 1
    MIN = 2
    COUNT = 3
    SUM = 4
    AVERAGE = 5


class _Operator(enum.Enum):
    """The boolean operators used by WikiSQL. Indexes match the data."""

    EQUALS = 0
    GREATER = 1
    LESSER = 2


@dataclasses.dataclass
class _Condition:
    """Represents an SQL where clauses (e.g A = "a" or B > 5)."""

    column: Text
    operator: _Operator
    cmp_value: Any


_TOKENIZER = re.compile(r"\w+|[^\w\s]+", re.UNICODE | re.MULTILINE | re.DOTALL)


def _normalize_for_match(x):
    return list(_TOKENIZER.findall(x.lower()))


def _compare(operator, src, tgt):
    if operator == _Operator.EQUALS:
        return src == tgt
    elif operator == _Operator.GREATER:
        return src > tgt
    elif operator == _Operator.LESSER:
        return src < tgt
    raise ValueError(f"Unknown operator: {operator}")


def _parse_value(table, column, cell_value):
    """Convert numeric values to floats and keeps everything else as string."""
    types = table["types"]
    return _TYPE_CONVERTER[types[column]](cell_value)


def _is_string(x):
    return isinstance(x, str)


def _respect_conditions(table, row, conditions):
    """True if 'row' satisfies all 'conditions'."""
    for cond in conditions:
        table_value = row[cond.column]

        cmp_value = _parse_value(table, cond.column, cond.cmp_value)

        if _is_string(table_value) and _is_string(cmp_value):
            table_value = _normalize_for_match(table_value)
            cmp_value = _normalize_for_match(cmp_value)

        if not isinstance(table_value, type(cmp_value)):
            raise ValueError("Type difference {} != {}".format(type(table_value), type(cmp_value)))

        if not _compare(cond.operator, table_value, cmp_value):
            return False
    return True


def _get_float_answer(table, answer_coordinates, aggregation_op):
    """Applies operation to produce reference float answer."""
    if not answer_coordinates:
        if aggregation_op == _Aggregation.COUNT:
            return 0.0
        else:
            return EMPTY_ANSWER_AGG

    # Count can support non numeric answers.
    if aggregation_op == _Aggregation.COUNT:
        return float(len(answer_coordinates))

    # If we have just one answer, if float returns it or try a conversion.
    values = [table["rows"][i][j] for (i, j) in answer_coordinates]
    if len(answer_coordinates) == 1:
        try:
            return convert_to_float(values[0])
        except ValueError as e:
            if aggregation_op != _Aggregation.NONE:
                raise e

    if aggregation_op == _Aggregation.NONE:
        return None

    # Other aggregation only support numeric values. Bail out if we have strings.
    if not all((isinstance(v, (int, float)) for v in values)):
        return None

    if aggregation_op == _Aggregation.SUM:
        return float(sum(values))
    elif aggregation_op == _Aggregation.AVERAGE:
        return sum(values) / len(answer_coordinates)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation_op}")


def _get_answer_coordinates(table, sql_query):
    """Retrieves references coordinates by executing SQL."""
    # MAX and MIN are automatically supported by the model.
    aggregation_op_index = sql_query["agg"]
    if aggregation_op_index >= 3:
        aggregation_op = _Aggregation(aggregation_op_index)
    else:
        aggregation_op = _Aggregation.NONE

    target_column = sql_query["sel"]
    conditions = [
        _Condition(column, _Operator(operator), cmp_value)
        for column, operator, cmp_value in zip(
            sql_query["conds"]["column_index"], sql_query["conds"]["operator_index"], sql_query["conds"]["condition"]
        )
    ]

    indices = []
    for row in range(len(table["rows"])):
        if _respect_conditions(table, table["rows"][row], conditions):
            indices.append((row, target_column))

    if not indices:
        return [], aggregation_op

    if len(indices) == 1:
        return indices, aggregation_op

    # Parsing of MIN/MAX.
    if aggregation_op_index in (1, 2):
        operators = {2: min, 1: max}
        values = [(table["rows"][i][j], index) for index, (i, j) in enumerate(indices)]
        reduced = functools.reduce(operators[sql_query["agg"]], values)

        ret = [indices[reduced[1]]]
        return ret, _Aggregation.NONE

    return indices, aggregation_op


def _get_answer_text(table, answer_coordinates, float_answer):
    if float_answer is not None:
        return [str(float_answer)]
    return [str(table["real_rows"][r][c]) for r, c in answer_coordinates]


def retrieve_wikisql_query_answer_tapas(table, example) -> List:
    answer_coordinates, aggregation_op = _get_answer_coordinates(table, example)
    float_answer = _get_float_answer(table, answer_coordinates, aggregation_op)
    answer_text = _get_answer_text(table, answer_coordinates, float_answer)
    # keep the original data the same with TaPas
    if len(answer_text) == 0:
        answer_text = [EMPTY_ANSWER]
    return answer_text


def preprocess_tableqa_function3(examples):

        def transform_condition(input_dict):

            column_indexes = input_dict.get('column_index', [])
            operator_indexes = input_dict.get('operator_index', [])
            conditions = input_dict.get('condition', [])    
            transformed_output = [[col_idx, op_idx, cond] for col_idx, op_idx, cond in zip(column_indexes, operator_indexes, conditions)]

            return transformed_output


        def _convert_table_types(_table):
            """Runs the type converter over the table cells."""
            ret_table = deepcopy(_table)
            types = ret_table["types"]
            ret_table["real_rows"] = ret_table["rows"]
            typed_rows = []
            for row in ret_table["rows"]:
                typed_row = []
                for column, cell_value in enumerate(row):
                    typed_row.append(_TYPE_CONVERTER[types[column]](cell_value))
                typed_rows.append(typed_row)
            ret_table["rows"] = typed_rows
            return ret_table

        questions = [question.lower() for question in examples["question"]]
        example_table = examples["table"]
        example_sql = examples["sql"]
        table = pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
            
        tapas_table = _convert_table_types(example_table)
        answer_list: List[str] = retrieve_wikisql_query_answer_tapas(tapas_table, example_sql)


        sql = Query(examples['sql']['sel'], examples['sql']['agg'], transform_condition(examples['sql']['conds'])).__repr__()
        
        return {'answers':", ".join(answer_list),
                 "sql_clean":sql}
    




class Query:

    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    syms = ['SELECT', 'WHERE', 'AND', 'COL', 'TABLE', 'CAPTION', 'PAGE', 'SECTION', 'OP', 'COND', 'QUESTION', 'AGG', 'AGGOPS', 'CONDOPS']

    def __init__(self, sel_index, agg_index, conditions=tuple(), ordered=False):
        self.sel_index = sel_index
        self.agg_index = agg_index
        self.conditions = list(conditions)
        self.ordered = ordered

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            indices = self.sel_index == other.sel_index and self.agg_index == other.agg_index
            if other.ordered:
                conds = [(col, op, str(cond).lower()) for col, op, cond in self.conditions] == [(col, op, str(cond).lower()) for col, op, cond in other.conditions]
            else:
                conds = set([(col, op, str(cond).lower()) for col, op, cond in self.conditions]) == set([(col, op, str(cond).lower()) for col, op, cond in other.conditions])

            return indices and conds
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __repr__(self):
        rep = 'SELECT {agg} {sel} FROM table'.format(
            agg=self.agg_ops[self.agg_index],
            sel='col{}'.format(self.sel_index),
        )
        if self.conditions:
            rep +=  ' WHERE ' + ' AND '.join(['{} {} "{}"'.format('col{}'.format(i), self.cond_ops[o], v) for i, o, v in self.conditions])
        return rep

    