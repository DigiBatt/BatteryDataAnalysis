import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cmcrameri import cm
import matplotlib.pyplot as plt

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
# colorscale_dict = {'C': [[0.0, "#A3D8FF"],[0.5, "#3399FF"],[1.0, "#003366"]], 'D': [[0.0, "#FF9999"],[0.5, "#FF3333"],[1.0, "#CC0000"]]}
# colorscale_dict = {'C': 'Blues', 'D': 'Reds'}

def get_colorscale(state, nb_cycle, i):
    colorscale_dict = {'C': 'Blues', 'D': 'Reds'}
    colorscale = colorscale_dict[state]
    color_value = (i+1) / (nb_cycle + 2)
    cmap = plt.get_cmap(colorscale)
    rgba_color = cmap(color_value)
    hex_color = f'rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {rgba_color[3]})'

    return hex_color
    
def plot_DQDV_result(df,
                     file_path,
                     save=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    fig_DQDV = go.Figure()
    nb_cycle = df['Cycle'].max()
    for state in ['C', 'D']:
        for i, cycle in enumerate(df['Cycle'].unique()):
            df_state = df[(df['State'] == state) & (df['Cycle'] == cycle)].sort_values(by='smoothed_voltage')
            if len(df_state) > 0:

                hex_color = get_colorscale(state, nb_cycle, i)
                fig_DQDV.add_trace(go.Scatter(
                    x=df_state['smoothed_voltage'],
                    y=df_state['smoothed_dqdv'],
                    mode='lines+markers',
                    marker=dict(
                        color=hex_color
                        ),
                    hovertext = f'Cycle: {cycle}<br>'
                    +'State: ' + df_state['State'].astype(str),
                    name=f'Cycle {cycle}, State {state}',
                    ))
    fig_DQDV.update_layout(
        title='Parameter numerical results<br>'
        f'File : <b>{file_name}</b>',
        xaxis_title="Voltage (V)",
        yaxis_title='dQ/dV (Ah/V)',
        ) 
    
    if save:
        file_path_DQDV = os.path.join(folder_path, f'DQDV_{file_name}.html')
        fig_DQDV.write_html(file_path_DQDV)
    
    return fig_DQDV

# def plot_dqdv(df,
#               file_path,
#               save=False):
#     file_name = os.path.splitext(os.path.basename(file_path))[0]
#     folder_path = os.path.dirname(file_path)

#     fig_dqdv = go.Figure()
#     fig_dqdv.add_trace(go.Scatter(
#         x=df["smoothed_voltage"],
#         y=df["smoothed_dqdv"],
#         mode='markers',
#         marker=dict(
#             size=4,
#             color=df["Cycle"],
#             colorscale=cycle_colorscale,
#             showscale=False,
#         ),
#         hovertext=[f'Cycle: {cycle}' for cycle in df["Cycle"]],
#         ))
    
#     fig_dqdv.update_layout(
#         title='dQ/dV curves over every cycles<br>'
#         f'File : <b>{os.path.basename(file_path)}</b>',
#         title_font=dict(size=16),
#         xaxis_title="Voltage",
#         yaxis_title="dQ/dV",
#         )
    
#     if save:
#         file_path_dqdv = os.path.join(folder_path, f'dqdv_{file_name}.html')
#         fig_dqdv.write_html(file_path_dqdv)

#     return fig_dqdv


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
            title="Voltage (V)",
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
        xaxis_title="Capacity (Ah)",
        yaxis_title="Voltage (V)",
        )
    
    if save:
        file_path_pocv = os.path.join(folder_path, f'pocv_{file_name}.html')
        fig_pocv.write_html(file_path_pocv)

    return fig_pocv


def reduce_points_number(df_input, n_cycle):
    N = max(int(len(df_input) / (n_cycle * len(df_input['Cycle'].unique()))), 1)

    return df_input.iloc[::N]


def plot_GITT_test(df_nested,
              file_path,
              save=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    df = pd.concat(df_nested.values())

    N = max(1, int(len(df) / 100000))
    df = df.iloc[::N]

    fig_GITT = go.Figure()
    fig_GITT.add_trace(go.Scatter(
        x=df['TestTime'],
        y=df["Voltage"],
        hovertext='Pulse: ' + df['Pulse'].astype(str)
        + '<br>Relaxation: ' + df['Relaxation'].astype(str)
        + '<br>Cycle: ' + df['Cycle'].astype(str)
        + '<br>State: ' + df['State'].astype(str),
        name='Voltage (V)',
        ))
    fig_GITT.add_trace(go.Scatter(
        x=df['TestTime'],
        y=df["Current"],
        hovertext='Pulse: ' + df['Pulse'].astype(str) 
        + '<br>Relaxation: ' + df['Relaxation'].astype(str)
        + '<br>Cycle: ' + df['Cycle'].astype(str)
        + '<br>State: ' + df['State'].astype(str),
        name='Current (A)',
    ))
    fig_GITT.add_trace(go.Scatter(
        x=df['TestTime'],
        y=df["Capacity"],
        hovertext='Pulse: ' + df['Pulse'].astype(str) 
        + '<br>Relaxation: ' + df['Relaxation'].astype(str)
        + '<br>Cycle: ' + df['Cycle'].astype(str)
        + '<br>State: ' + df['State'].astype(str),
        name='Capacity (Ah)',
    ))
    fig_GITT.update_layout(
        title="Voltage, Current and Capacity over TestTime<br>"
        f'File : <b>{file_name}</b><br>',
        xaxis_title="TestTime (s)",
        yaxis_title="Voltage, Current and Capacity",
        )
    
    if save:
        file_path_GITT = os.path.join(folder_path, f'GITT_{file_name}.html')
        fig_GITT.write_html(file_path_GITT)
    
    return fig_GITT


def plot_GITT_result(df,
                     file_path,
                     column='Diffusion Coefficient',
                     save=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    fig_GITT = go.Figure()
    nb_cycle = df['Cycle'].max()
    for state in ['C', 'D']:
        for i, cycle in enumerate(df['Cycle'].unique()):
            df_state = df[(df['State'] == state) & (df['Cycle'] == cycle)].sort_values(by='SOC')

            hex_color = get_colorscale(state, nb_cycle, i)

            fig_GITT.add_trace(go.Scatter(
                x=df_state['SOC'],
                y=df_state[column],
                mode='lines+markers',
                line=dict(color=hex_color,
                    ),
                hovertext = f'Cycle: {cycle}<br>'
                +'Pulse: ' + df_state['Pulse'].astype(str),
                name=f'Cycle {cycle}, State {state}',
                ))
    fig_GITT.update_layout(
        title='Parameter numerical results<br>'
        f'File : <b>{file_name}</b>',
        xaxis_title="SOC (%)",
        yaxis_title=column,
        )
    
    if save:
        file_path_GITT = os.path.join(folder_path, f'GITT_{file_name}.html')
        fig_GITT.write_html(file_path_GITT)
    
    return fig_GITT


def plot_HPPC_test(df_nested,
                   file_path,
                   save=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    df = pd.concat(df_nested.values())

    N = max(1, int(len(df) / 100000))
    df = df.iloc[::N]

    fig_HPPC = go.Figure()
    fig_HPPC.add_trace(go.Scatter(
        x=df['TestTime'],
        y=df["Voltage"],
        hovertext='Pulse: ' + df['Pulse'].astype(str)
        + '<br>Cycle: ' + df['Cycle'].astype(str)
        + '<br>State: ' + df['State'].astype(str),
        name='Voltage (V)',
        ))
    fig_HPPC.add_trace(go.Scatter(
        x=df['TestTime'],
        y=df["Current"],
        hovertext='Pulse: ' + df['Pulse'].astype(str)
        + '<br>Cycle: ' + df['Cycle'].astype(str)
        + '<br>State: ' + df['State'].astype(str),
        name='Current (A)',
    ))
    fig_HPPC.add_trace(go.Scatter(
        x=df['TestTime'],
        y=df["Capacity"],
        hovertext='Pulse: ' + df['Pulse'].astype(str)
        + '<br>Cycle: ' + df['Cycle'].astype(str)
        + '<br>State: ' + df['State'].astype(str),
        name='Capacity (Ah)',
    ))
    fig_HPPC.update_layout(
        title="Voltage, Current and Capacity over TestTime<br>"
        f'File : <b>{file_name}</b><br>',
        xaxis_title="TestTime (s)",
        yaxis_title="Voltage, Current and Capacity",
        )
    
    if save:
        file_path_HPPC = os.path.join(folder_path, f'HPPC_{file_name}.html')
        fig_HPPC.write_html(file_path_HPPC)
    
    return fig_HPPC


def plot_HPPC_result(df,
                     file_path,
                     column='R',
                     save=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    fig_HPPC = go.Figure()
    nb_cycle = df['Cycle'].max()
    for state in ['C', 'D']:
        for i, cycle in enumerate(df['Cycle'].unique()):
            df_state = df[(df['State'] == state) & (df['Cycle'] == cycle)].sort_values(by='SOC')

            for charge in ['charge', 'discharge']:
                hex_color = get_colorscale('C', nb_cycle, i) if charge == 'charge' else get_colorscale('D', nb_cycle, i)

                fig_HPPC.add_trace(go.Scatter(
                    x=df_state['SOC'],
                    y=df_state[f'{column}_{charge}'],
                    mode='lines+markers',
                    line=dict(color=hex_color,
                            ),
                    hovertext = f'Cycle: {cycle}<br>'
                    +'Pulse: ' + df_state['Pulse'].astype(str),
                    name=f'Cycle {cycle}, {charge} pulse',
                    ))
    fig_HPPC.update_layout(
        title='Parameter numerical results<br>'
        f'File : <b>{file_name}</b>',
        xaxis_title="SOC (%)",
        yaxis_title=column,
        )
    
    if save:
        file_path_HPPC = os.path.join(folder_path, f'HPPC_{file_name}.html')
        fig_HPPC.write_html(file_path_HPPC)
    
    return fig_HPPC