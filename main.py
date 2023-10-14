import math

import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.plotly as py
from igraph import *


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

def visualise(v_num):
    v_label = map(str, range(v_num))
    G = Graph.Tree(v_num, 10)
    lay = G.layout('rt')

    position = {k: lay[k] for k in range(v_num)}
    Y = [lay[k][1] for k in range(v_num)]
    M = max(Y)

    es = EdgeSeq(G)  # sequence of edges
    E = [e.tuple for e in G.es]  # list of edges

    L = len(position)
    Xn = [position[k][0] for k in range(L)]
    Yn = [2 * M - position[k][1] for k in range(L)]
    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

    labels = v_label

    # Create Plotly Traces

    lines = go.Scatter(x=Xe,
                       y=Ye,
                       mode='lines',
                       line=dict(color='rgb(210,210,210)', width=1),
                       hoverinfo='none'
                       )
    dots = go.Scatter(x=Xn,
                      y=Yn,
                      mode='markers',
                      name='',
                      marker=dict(symbol='dot',
                                  size=18,
                                  color='#6175c1',  # '#DB4551', 
                                  line=dict(color='rgb(50,50,50)', width=1)
                                  ),
                      text=labels,
                      hoverinfo='text',
                      opacity=0.8
                      )

    # Create Text Inside the Circle via Annotations

    def make_annotations(pos, text, font_size=10,
                         font_color='rgb(250,250,250)'):
        L = len(pos)
        if len(text) != L:
            raise ValueError('The lists pos and text must have the same len')
        annotations = go.Annotations()
        for k in range(L):
            annotations.append(
                go.Annotation(
                    text=labels[k],  # or replace labels with a different list 
                    # for the text within the circle  
                    x=pos[k][0], y=2 * M - position[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
        return annotations

        # Add Axis Specifications and Create the Layout

    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )

    layout = dict(title='Tree with Reingold-Tilford Layout',
                  annotations=make_annotations(position, v_label),
                  font=dict(size=12),
                  showlegend=False,
                  xaxis=go.XAxis(axis),
                  yaxis=go.YAxis(axis),
                  margin=dict(l=40, r=40, b=85, t=100),
                  hovermode='closest',
                  plot_bgcolor='rgb(248,248,248)'
                  )

    # Plot

    data = go.Data([lines, dots])
    fig = dict(data=data, layout=layout)
    fig['layout'].update(annotations=make_annotations(position, v_label))
    py.iplot(fig, filename='Tree-Reingold-Tilf')

if __name__ == '__main__':
    # data, outcome = prepare_data()
    # build_tree(data, outcome)
    visualise(25)
