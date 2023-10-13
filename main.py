import math

import pandas as pd


def single_entropy(value:int, size:int):
    if value == 0:
        return 0
    return -1*(value/size)*math.log2((value/size))
def entropy(data:pd.DataFrame, outcome:pd.Series, column:str):
    used_data = data[column]
    used_data.name = column
    size = used_data.size
    used_data = pd.concat([used_data, outcome], axis=1)
    used_data = used_data.groupby(column)['outcome'].sum()
    used_data['entropy'] = used_data.apply(lambda row: single_entropy(row, size))
    return used_data['entropy'].sum(), used_data

def conditional_entropy(data):
    print(data)

def mapper(age:int):
    if age > 40:
        return "old"
    elif age > 20:
        return 'medium'
    else:
        return "young"


def map_age(data:pd.DataFrame):
    data['Age'] = data.apply(lambda row: mapper(row['Age']), axis=1)
    return data

def prepare_data():
    data = pd.read_csv("data/titanic-homework.csv", index_col="PassengerId")
    outcome = data['Survived']
    outcome.name = "outcome"
    data = data.drop(['Name', 'Survived'], axis='columns')
    data = map_age(data)
    return data, outcome

if __name__ == '__main__':
    data, outcome = prepare_data()
    for col in data.columns:
        ent, aggr = entropy(data, outcome, col)
        print(f"{col}: {ent}")
        conditional_entropy(aggr)



