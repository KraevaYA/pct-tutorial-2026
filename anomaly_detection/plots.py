import numpy as np
import pandas as pd
from itertools import groupby
from operator import itemgetter

# for visualization
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
import plotly.express as px
plotly.offline.init_notebook_mode(connected=True)

from anomaly_detection.metrics import _get_discords_errors


def plot_ts(ts: pd.Series,
            label: str = None,
            title: str = None,
            x_label: str = None,
            y_label: str = None) -> None:
    """
    Plot the time series

    Parameters
    ----------
    ts : pandas.Series
        Time series.

    label : str, default = None
        Name of time series.

    title : str, default = None
        Title of plot.
    
    x_label : str, default = None
        Title of x-axis.

    y_label : str, default = None
        Title of y-axis.
    """

    n = len(ts)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, line=dict(width=3), name=label))

    fig.update_xaxes(showgrid=False,
                     title=x_label,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickformat="%Y-%m-%d %H:%M",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title=y_label,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18, color='black'),
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title=title,
                      title_font=dict(size=24, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=20, color='black')),
                      width=1500,
                      height=400
                      )

    fig.show(renderer="colab")


def plot_discords(ts: pd.DataFrame, discords: dict, is_detailed: bool = False) -> None:
    """
    Visualize the plot that includes the time series with top-k discords, annotation, and matrix profile

    Parameters
    ----------
    ts : pandas.DataFrame
        Time series.
    
    discords : dict
        Top-k discords.
    
    is_detailed : bool, default = False
        If is_detailed = True, the TP, FP, FN discords are visualized in the plot with time series. 
    """

    top_k = len(discords['indices'])

    plot_num = 2
    if (len(ts.shape) != 1):
        plot_num += 1
    else:
        ts = ts.to_frame()
        
    fig = make_subplots(rows=plot_num, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.02)
    
    # plot time series with discords
    fig.add_trace(go.Scatter(x=ts.index, 
                             y=ts['value'], 
                             line=dict(color='#636EFA'), name="Time Series"), 
                  row=1, col=1)

    for i in range(top_k):
        discord_idx = discords['indices'][i]
        discord_m = discords['m'][i]
        fig.add_trace(go.Scatter(x=ts.index[np.arange(discord_idx, discord_idx+discord_m-1)], 
                                 y=ts['value'][discord_idx:discord_idx+discord_m-1], 
                                 line=dict(color='red'), name=f"Top-{i+1} discord"), 
                      row=1, col=1)
        
    if is_detailed:

        _, discords_errors = _get_discords_errors(ts['label'].values, discords['indices'], discords['m'], full_report = True)

        type_colors = {'TP': 'green', 'FP': 'grey', 'FN': 'red'}
        for error_type, discords_list in discords_errors.items():
            for idx, m in discords_list:
                fig.add_vrect(x0=ts.index[idx], x1=ts.index[idx+m-1],
                        fillcolor=type_colors[error_type], opacity=0.15, line_width=0,
                        row=1, col=1)
                fig.add_annotation(
                    x=ts.index[int((2*idx+m)/2)],  
                    y=1.02,
                    xref="x",
                    yref="paper",
                    text=error_type,
                    showarrow=False,
                    font=dict(size=18, color="black"),
                    xanchor="center",
                    yanchor="bottom"
                    )

    # plot matrix profile
    indices = np.where(discords['mp'] == -np.inf)[0]
    discords['mp'][indices] = 0

    discords_idx = discords['indices']
    discords_mp = [discords['mp'][idx] for idx in discords_idx]

    fig.add_trace(go.Scatter(x=ts.index, y=discords['mp'], line=dict(color='#636EFA', width=2), name="Matrix Profile"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts.index[discords_idx], y=discords_mp, mode='markers', marker=dict(symbol='star', color='red', size=7), name="Discords"), row=2, col=1)

    #plot annotation
    if ts.shape[1] > 1:
        fig.add_trace(go.Scatter(x=ts.index, y=ts['label'], line=dict(color='green'), name="Annotation"), row=3, col=1)

    fig.update_layout(title_text="Top-k discords in time series")

    fig.update_xaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickformat="%Y-%m-%d %H:%M",
                     tickfont=dict(size=18, color='black'),
                     linewidth=2,
                     tickwidth=2)
    fig.update_yaxes(showgrid=False,
                     title_font=dict(size=22, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=18), color='black',
                     zeroline=False,
                     linewidth=2,
                     tickwidth=2)

    fig.update_layout(title_font=dict(size=24, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      width=1500)

    fig.show(renderer="colab")


def plot_heatmap(ts, discords: dict):
    """
    Visualize the discord heatmap.

    Parameters
    ----------
    ts : pandas.DataFrame
        Time series.
    
    discords : dict
        Top-k discords.
    """
    
    red_colors = [px.colors.sequential.Reds[8], px.colors.sequential.Reds[7], px.colors.sequential.Reds[6], 
                  px.colors.sequential.Reds[5], px.colors.sequential.Reds[3], px.colors.sequential.Reds[2]]

    minL = int(list(discords.keys())[0])
    maxL = int(list(discords.keys())[-1])
    n = len(discords[str(minL)]['mp'])

    heatmap_values = np.full((maxL-minL+1, n), -np.inf)
    i = 0
    for m in range(minL, maxL+1):
        heatmap_values[i][:(n-m+1)] = list(map(lambda item: (item**2) /(2*m), discords[str(m)]['mp'][0:(n-m+1)]))
        i = i + 1

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
                   z = heatmap_values,
                   y = np.arange(minL, maxL+1),
                   x = ts.index,
                   hoverongaps = False,
                   colorscale = [[0, 'rgb(255,255,255, 1)'],
                                  [0.05, 'rgb(255,255,255, 1)'],
                                  [0.05, px.colors.sequential.Reds[0]],
                                  [0.24, px.colors.sequential.Reds[2]],
                                  [0.43, px.colors.sequential.Reds[3]],
                                  [0.62, px.colors.sequential.Reds[5]],
                                  [0.81, px.colors.sequential.Reds[6]],
                                  [1, px.colors.sequential.Reds[8]]
                   ],
                   colorbar = dict(thickness=20, outlinecolor='black', outlinewidth=0.5),
                   zmin=0, zmax=1), 
                   )

    fig.update_layout(title="<b>Discord Heatmap</b>",
                      title_font=dict(size=22, color='black'),
                      margin=dict(l=10, r=10, b=10, t=50), 
                      plot_bgcolor="#fff", 
                      width=1500, 
                      height=400, 
                      coloraxis_showscale=False, 
                      showlegend=False)

    fig.update_xaxes(showgrid=False, 
                     linecolor='#000', 
                     mirror=True, 
                     ticks="outside", 
                     tickfont=dict(size=18, color='black'),
                     ) 
    fig.update_yaxes(title='Discord lengths',
                     title_font=dict(size=22, color='black'),
                     showgrid=False, 
                     linecolor='#000', 
                     mirror=True, 
                     ticks="outside", 
                     tickfont=dict(size=18, color='black'), 
                     zeroline=False
                     )
                  
    fig.show(renderer="colab")
