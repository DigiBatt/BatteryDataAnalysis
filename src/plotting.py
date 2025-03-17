import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cmcrameri import cm
import matplotlib.pyplot as plt
from IPython.display import Image, display

cycle_colorscale = [[0.0, "#A3D8FF"], [0.5, "#3399FF"], [1.0, "#003366"]]

def matplotlib_to_plotly_colorscale(colormap, n_colors=256):
    colors = [colormap(i / (n_colors - 1)) for i in range(n_colors)]
    # colorscale = [(0, f"rgb({int(colors[0][0] * 255)}, {int(colors[0][1] * 255)}, {int(colors[0][2] * 255)})")]
    # colorscale += [
    #     (i / (2*(n_colors - 1)) + 0.5, f"rgb({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})")
    #     for i, c in enumerate(colors)]
    
    colorscale = [(i / (n_colors - 1), f"rgb({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})")for i, c in enumerate(colors)]
    return colorscale

batlow_colorscale = matplotlib_to_plotly_colorscale(cm.lipari)

def get_colorscale(state, nb_cycle, i):
    colorscale_dict = {'C': 'Blues', 'D': 'Reds'}
    colorscale = colorscale_dict[state]
    color_value = (i+1) / (nb_cycle + 2)
    cmap = plt.get_cmap(colorscale)
    rgba_color = cmap(color_value)
    hex_color = f'rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {rgba_color[3]})'

    return hex_color

title_size = 45 # [20,45]
axis_size = 35 # [18,35]
legend_size = 35 # [18,35]
tickfont_size = 35 # [18,35]
marker_size = 10
line_width = 3


def plot_test_over_time(df_input,
                        file_path,
                        test='CCCV',
                        pulse=False,
                        save=False,
                        png=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    param_dict = {'Voltage': 'Voltage  /  V', 
                  'Current': 'Current  /  A', 
                  'Capacity': 'Capacity  /  Ah',}

    if pulse:
        df = pd.concat(df_input.values())
        N = max(1, int(len(df) / 100000))
        df = df.iloc[::N]
        hovertext = ('Pulse: ' + df['Pulse'].astype(str)
                     + '<br>Cycle: ' + df['Cycle'].astype(str) 
                     + '<br>State: ' + df['State'].astype(str)
                     + '<br>C_Rate: ' + df['C_Rate'].astype(str))
    else:
        df = df_input.copy()
        N = max(1, int(len(df) / 100000))
        df = df.iloc[::N]
        hovertext = ('Cycle: ' + df['Cycle'].astype(str) 
                     + '<br>State: ' + df['State'].astype(str)
                     + '<br>C_Rate: ' + df['C_Rate'].astype(str)) 

    fig = go.Figure()
    for param in param_dict.keys():
        fig.add_trace(go.Scatter(
            x=df['TestTime'],
            y=df[param],
            mode='lines',
            line=dict(width=line_width),
            text=hovertext,
            name=param_dict[param],
            ))
    
    fig.update_layout(
        title={
            'text': f'<b>Voltage Current and Capacity for {test} Test</b><br>'
            f'File : {file_name}',
            'font': {'size': title_size}
        },
        xaxis_title="<b>TestTime  /  s</b>",
        yaxis_title='<b>Voltage, Current and Capacity</b>',
        xaxis={'title': {'font': {'size': axis_size}},
               'tickfont': {'size': tickfont_size}},
        yaxis={'title': {'font': {'size': axis_size}},
               'tickfont': {'size': tickfont_size}},
        legend={'font': {'size': legend_size}}
        )
    
    if save:
        file_path = os.path.join(folder_path, f'{test}_{file_name}.html')
        fig.write_html(file_path)
    if png:
        png_image = fig.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    else:
        fig.show()

    return fig


def plot_DQDV_result(df,
                     file_path,
                     save=False,
                     png=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    fig_dqdv = go.Figure()
    nb_cycle = df['Cycle'].max()
    for state in ['C', 'D']:
        for i, cycle in enumerate(df['Cycle'].unique()):
            df_state = df[(df['State'] == state) & (df['Cycle'] == cycle)].sort_values(by='smoothed_voltage')
            if len(df_state) > 0:

                hex_color = get_colorscale(state, nb_cycle, i)
                fig_dqdv.add_trace(go.Scatter(
                    x=df_state['smoothed_voltage'],
                    y=df_state['smoothed_dqdv'],
                    mode='lines+markers',
                    marker=dict(
                        color=hex_color,
                        size=marker_size,
                        ),
                    line=dict(width=line_width),
                    hovertext = f'Cycle: {cycle}<br>'
                    +'State: ' + df_state['State'].astype(str),
                    name=f'Cycle {cycle}, State {state}',
                    ))
    
    fig_dqdv.update_layout(
        title={
            'text': f'<b>dQ/dV curves over cycles</b><br>',
            # f'<i>File : {file_name}</i>',
            'font': {'size': title_size}
        },
        xaxis_title="<b>Voltage  /  V</b>",
        yaxis_title="<b>dQ/dV  /  Ah/V</b>",
        xaxis={'title': {'font': {'size': axis_size}},
               'tickfont': {'size': tickfont_size}},
        yaxis={'title': {'font': {'size': axis_size}},
               'tickfont': {'size': tickfont_size}},
        legend={'font': {'size': legend_size}}
        )
    
    if save:
        file_path_dqdv = os.path.join(folder_path, f'DQDV_{file_name}.html')
        fig_dqdv.write_html(file_path_dqdv)
    if png:
        png_image = fig_dqdv.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    else:
        fig_dqdv.show()
    
    return fig_dqdv


def plot_dqdv_heatmap(df,
                      file_path,
                      save=False,
                      png=False):
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
            # colorbar_title=colorbar_title,
            colorbar=dict(
                title=dict(
                    text=colorbar_title,
                    font=dict(size=legend_size)
                ),
            yanchor=yanchor,
            len=0.5,
        ),
            ),
            row=i_charge + 1, col=1
        )
        fig_dqdv.update_xaxes(
            title="Voltage (V)",
            title_font=dict(size=axis_size),
            row=i_charge + 1, col=1
            )
        fig_dqdv.update_yaxes(
            title="Cycle",
            title_font=dict(size=axis_size),
            row=i_charge + 1, col=1
            )
        
    fig_dqdv.update_layout(
        title="dQ/dV heatmap showing its magnitude's evolution through cycles<br>",
        # f'File : {file_name}',
        title_font=dict(size=title_size),
        font=dict(size=20),
        )

    if save:
        file_path_heatmap = os.path.join(folder_path, f'dqdv_heatmap_{file_name}.html')
        fig_dqdv.write_html(file_path_heatmap)
    if png:
        png_image = fig_dqdv.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    else:
        fig_dqdv.show()

    return fig_dqdv


def plot_pocv(df,
              file_path,
              save=False,
              png=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    df = reduce_points_number(df, n_cycle=1000)

    fig_pocv = go.Figure()
    nb_cycle = len(df['Cycle'].unique())

    for state in ['C', 'D']:
        for i, cycle in enumerate(df['Cycle'].unique()):
            df_state = df[(df['State'] == state) 
                          & (df['Cycle'] == cycle) 
                          & (df['normcurrent'] != 0)].sort_values(by='Voltage')
            if len(df_state) > 0:

                hex_color = get_colorscale(state, nb_cycle, i)
                fig_pocv.add_trace(go.Scatter(
                    x=df_state['Capacity'],
                    y=df_state['Voltage'],
                    mode='lines+markers',
                    marker=dict(
                        color=hex_color,
                        size=marker_size,),
                    line=dict(width=3),
                    hovertext = f'Cycle: {cycle}<br>'
                    +'State: ' + df_state['State'].astype(str)
                    +'C_Rate: ' + df_state['C_Rate'].astype(str),
                    name=f'Cycle {cycle}, State {state}',
                    ))
    
    fig_pocv.update_layout(
        title={
            'text': f'<b>POCV curve over cycles</b><br>',
            # f'File : {file_name}',
            'font': {'size': title_size}
        },
        xaxis_title="<b>Capacity  /  Ah</b>",
        yaxis_title="<b>Voltage  /  V</b>",
        xaxis={'title': {'font': {'size': axis_size}},
               'tickfont': {'size': tickfont_size}},
        yaxis={'title': {'font': {'size': axis_size}},
               'tickfont': {'size': tickfont_size}},
        legend={'font': {'size': legend_size}}
        )
    
    if save:
        file_path_pocv = os.path.join(folder_path, f'pocv_{file_name}.html')
        fig_pocv.write_html(file_path_pocv)
    if png:
        png_image = fig_pocv.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    else:
        fig_pocv.show()

    return fig_pocv


def reduce_points_number(df_input, n_cycle):
    N = max(int(len(df_input) / (n_cycle * len(df_input['Cycle'].unique()))), 1)

    return df_input.iloc[::N]


def plot_GITT_result(df,
                     file_path,
                     column='Diffusion Coefficient',
                     save=False,
                     png=False):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    fig_GITT = go.Figure()
    nb_cycle = len(df['Cycle'].unique())
    for state in ['C', 'D']:
        for i, cycle in enumerate(df['Cycle'].unique()):
            df_state = df[(df['State'] == state) & (df['Cycle'] == cycle)].sort_values(by='SOC')

            hex_color = get_colorscale(state, nb_cycle, i)

            fig_GITT.add_trace(go.Scatter(
                x=df_state['SOC'],
                y=df_state[column],
                mode='lines+markers',
                marker=dict(color=hex_color,
                          size=marker_size),
                line=dict(width=line_width),
                hovertext = f'Cycle: {cycle}<br>'
                +'Pulse: ' + df_state['Pulse'].astype(str),
                name=f'Cycle {cycle}, State {state}',
                ))
    
    fig_GITT.update_layout(
        title={
            'text': f'<b>Parameter numerical results for GITT Test</b><br>'
            f'File : {file_name}',
            'font': {'size': title_size}
        },
        xaxis_title="SOC",
        yaxis_title=column,
        xaxis={'title': {'font': {'size': axis_size}},
               'tickfont': {'size': tickfont_size}},
        yaxis={'title': {'font': {'size': axis_size}},
               'tickfont': {'size': tickfont_size}},
        legend={'font': {'size': legend_size}}
        )
    
    if save:
        file_path_GITT = os.path.join(folder_path, f'GITT_{file_name}.html')
        fig_GITT.write_html(file_path_GITT)
    if png:
        png_image = fig_GITT.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    else:
        fig_GITT.show()
    
    return fig_GITT


def plot_HPPC_result(df,
                     file_path,
                     column='R',
                     save=False,
                     png=False):
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
                    marker=dict(color=hex_color,
                                size=marker_size,),
                    line=dict(width=line_width),
                    hovertext = f'Cycle: {cycle}<br>'
                    +'Pulse: ' + df_state['Pulse'].astype(str),
                    name=f'Cycle {cycle}, {charge} pulse',
                    ))
    
    fig_HPPC.update_layout(
        title={
            'text': f'<b>Parameter numerical results for HPPC Test</b><br>'
            f'File : {file_name}',
            'font': {'size': title_size}
        },
        xaxis_title="SOC",
        yaxis_title=column,
        xaxis={'title': {'font': {'size': axis_size}},
               'tickfont': {'size': tickfont_size}},
        yaxis={'title': {'font': {'size': axis_size}},
               'tickfont': {'size': tickfont_size}},
        legend={'font': {'size': legend_size}}
        )
    
    if save:
        file_path_HPPC = os.path.join(folder_path, f'HPPC_{file_name}.html')
        fig_HPPC.write_html(file_path_HPPC)
    if png:
        png_image = fig_HPPC.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    else:
        fig_HPPC.show()
    
    return fig_HPPC