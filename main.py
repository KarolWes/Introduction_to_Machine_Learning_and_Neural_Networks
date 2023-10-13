import math
import pandas as pd


def single_entropy(value: int, size: int):
    if value == 0:
        return 0
    return -1 * (value / size) * math.log2((value / size))


def entropy(data: pd.DataFrame, outcome: pd.Series, column: str):
    used_data = data[column]
    used_data.name = column
    size = used_data.size
    used_data = pd.concat([used_data, outcome], axis=1)
    positives = used_data.groupby(column)['outcome'].sum()
    counts = used_data.groupby(column)['outcome'].count()
    used_data = pd.concat([positives, counts], axis=1)
    used_data.columns = ['positives', 'count']
    used_data['entropy'] = used_data.apply(lambda row: single_entropy(row['positives'], size), axis=1)
    return used_data['entropy'].sum(), used_data


def conditional_entropy(data: pd.DataFrame):
    data["cond_ent"] = data.apply(lambda row: row["positives"] / row['count'] * row['entropy'], axis=1)
    return data['cond_ent'].sum(), data


def intrinsic(data: pd.DataFrame):
    data['intr'] = data.apply(lambda row: single_entropy(row['positives'], row['count']), axis=1)
    return data['intr'].sum(), data


# Legacy
def map_age_groups(age: int):
    if age > 40:
        return "old"
    elif age > 20:
        return 'medium'
    else:
        return "young"


def mapper(data: pd.DataFrame):
    data['Age'] = data.apply(lambda row: (
        "old" if row["Age"] > 40 else ("medium" if row["Age"] > 20 else "young")), axis=1)
    return data


def prepare_data():
    data = pd.read_csv("data/titanic-homework.csv", index_col="PassengerId")
    outcome = data['Survived']
    outcome.name = "outcome"
    data = data.drop(['Name', 'Survived'], axis='columns')
    data = mapper(data)
    return data, outcome


def best_branch(data, outcome):
    results = {}
    for col in data.columns:
        ent, aggr = entropy(data, outcome, col)
        # print(f"Entropy: {col}: {ent}")
        cond, aggr = conditional_entropy(aggr)
        # print(f"Conditional: {col}: {cond}")
        gain = ent - cond
        # print(f"Gain: {col}: {gain}")
        intr, aggr = intrinsic(aggr)
        # print(f"Intrinsic: {col}: {intr}")
        if intr != 0:
            ratio = gain / intr
        else:
            ratio = 0
        # print(f"Ratio: {ratio}")
        results[col] = ratio
    results = pd.DataFrame.from_dict(results, orient="index")
    results.columns = ["ratio"]
    results = results.sort_values("ratio", ascending=False)
    return results.head(1).index.values[0]


def purity_check(data, outcome, col):
    used_data = data[col]
    used_data.name = col
    used_data = pd.concat([used_data, outcome], axis=1)
    positives = used_data.groupby(col)['outcome'].sum()
    counts = used_data.groupby(col)['outcome'].count()
    used_data = pd.concat([positives, counts], axis=1)
    used_data.columns = ['positives', 'count']
    used_data['pure'] = used_data.apply(lambda row: row['positives'] == row['count'] or row['positives'] == 0, axis=1)
    purity = used_data['pure'].sum() == used_data.shape[0]
    return purity, used_data


def build_tree(data, outcome, depth=0):
    best = best_branch(data, outcome)
    purity, res = purity_check(data, outcome, best)
    for row in res.iterrows():
        print(depth*'\t' + best + ' - '+str(row[0])+': ')
        if not row[1]['pure']:
            new_data = data[data[best] == row[0]]
            new_data = new_data.drop(best, axis='columns')
            new_out = outcome[new_data.index]
            build_tree(new_data, new_out, depth+1)
        else:
            print((depth+1)*'\t'+f"decision:{1 if row[1]['positives'] > 0 else 0}")
    return



if __name__ == '__main__':
    data, outcome = prepare_data()
    build_tree(data, outcome)
