import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cmcrameri import cm

cycle_colorscale = [[0.0, "#A3D8FF"], [0.5, "#3399FF"], [1.0, "#003366"]]

def matplotlib_to_plotly_colorscale(colormap, n_colors=256):
    colors = [colormap(i / (n_colors - 1)) for i in range(n_colors)]
    colorscale = [(0, f"rgb({int(colors[0][0] * 255)}, {int(colors[0][1] * 255)}, {int(colors[0][2] * 255)})")]
    colorscale += [
        (i / (2*(n_colors - 1)) + 0.5, f"rgb({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})")
        for i, c in enumerate(colors)
        ]
    return colorscale

batlow_colorscale = matplotlib_to_plotly_colorscale(cm.lipari)

def plot_dqdv(df,
              file_path,
              save=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    fig_dqdv = go.Figure()
    fig_dqdv.add_trace(go.Scatter(
        x=df["smoothed_voltage"],
        y=df["smoothed_dqdv"],
        mode='markers',
        marker=dict(
            size=4,
            color=df["Cycle"],
            colorscale=cycle_colorscale,
            showscale=False,
        ),
        hovertext=[f'Cycle: {cycle}' for cycle in df["Cycle"]],
        ))
    
    fig_dqdv.update_layout(
        title='dQ/dV curves over every cycles<br>'
        f'File : <b>{os.path.basename(file_path)}</b>',
        title_font=dict(size=16),
        xaxis_title="Voltage",
        yaxis_title="dQ/dV",
        )
    
    if save:
        file_path_dqdv = os.path.join(folder_path, f'dqdv_{file_name}.html')
        fig_dqdv.write_html(file_path_dqdv)

    return fig_dqdv


def plot_dqdv_heatmap(df,
                      file_path,
                      save=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    fig_dqdv = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=['Charge', 'Discharge']
        )

    for i_charge, charging_state in enumerate(['C', 'D']):
        df_charge = df[df['State'] == charging_state]

        if charging_state == 'C':
            colorbar_title = 'log|dQ/dV| for <b>charge</b>'
            yanchor='bottom'
        else:
            colorbar_title = 'log|dQ/dV| for <b>discharge</b>'
            yanchor='top'

        fig_dqdv.add_trace(go.Heatmap(
            x=df_charge["smoothed_voltage"],
            y=df_charge["Cycle"],
            z=np.log(abs(df_charge["smoothed_dqdv"])),
            colorscale=batlow_colorscale,
            colorbar_title=colorbar_title,
            colorbar=dict(
                yanchor=yanchor,
                len=0.5,
                ),
            ),
            row=i_charge + 1, col=1
        )
        fig_dqdv.update_xaxes(
            title="Voltage",
            row=i_charge + 1, col=1
            )
        fig_dqdv.update_yaxes(
            title="Cycle",
            row=i_charge + 1, col=1
            )
        
    fig_dqdv.update_layout(
        title="dQ/dV heatmap showing its magnitude's evolution through cycles<br>"
        f'File : <b>{file_name}</b><br>',
        title_font=dict(size=16)
        )

    if save:
        file_path_heatmap = os.path.join(folder_path, f'dqdv_heatmap_{file_name}.html')
        fig_dqdv.write_html(file_path_heatmap)

    return fig_dqdv



def plot_pocv(df,
              file_path,
              save=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    df = reduce_points_number(df, n_cycle=1000)

    fig_pocv = go.Figure()
    fig_pocv.add_trace(go.Scatter(
        x=df["Capacity"],
        y=df["Voltage"],
        mode='markers',
        marker=dict(
            size=4,
            color=df["Cycle"],
            colorscale=cycle_colorscale,
            showscale=False,
        ),
        hovertext=[f'Cycle: {cycle}' for cycle in df["Cycle"]],
        ))

    fig_pocv.update_layout(
        title='Voltage over Capacity for smoothed data over every cycles<br>'
        f'File : <b>{file_name}</b>',
        xaxis_title="Capacity",
        yaxis_title="Voltage",
        )
    
    if save:
        file_path_pocv = os.path.join(folder_path, f'pocv_{file_name}.html')
        fig_pocv.write_html(file_path_pocv)

    return fig_pocv


def reduce_points_number(df_input, n_cycle):
    N = max(int(len(df_input) / (n_cycle * len(df_input['Cycle'].unique()))), 1)

    return df_input.iloc[::N]


def plot_GITT(df,
              file_path,
              save=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    fig_GITT = go.Figure()
    fig_GITT.add_trace(go.Scatter(
        x=df['TestTime'],
        y=df["Voltage/V"],
        hovertext='Pulse: ' + df['Pulse'].astype(str)
        + '<br>Relaxation: ' + df['Relaxation'].astype(str),
        name='Voltage',
        ))
    fig_GITT.add_trace(go.Scatter(
        x=df['TestTime'],
        y=df["Current/mA"],
        hovertext='Pulse: ' + df['Pulse'].astype(str) 
        + '<br>Relaxation: ' + df['Relaxation'].astype(str),
        name='Current',
    ))
    fig_GITT.update_layout(
        title="Voltage/V over time",
        xaxis_title="TestTime [√s]",
        yaxis_title="Voltage",
        )
    
    if save:
        file_path_GITT = os.path.join(folder_path, f'GITT_{file_name}.html')
        fig_GITT.write_html(file_path_GITT)
    
    return fig_GITT