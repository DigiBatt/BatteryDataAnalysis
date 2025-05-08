import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cmcrameri import cm
import matplotlib.pyplot as plt
from IPython.display import Image, display

title_size = 20  # [20,45]
axis_size = 18  # [18,35]
legend_size = 18  # [18,35]
tickfont_size = 18  # [18,35]
marker_size = 10
line_width = 3


def matplotlib_to_plotly_colorscale(colormap, n_colors=256):
    """Converts a matplotlib colormap to a Plotly colorscale

    Parameters
    ----------
    colormap : matplotlib colormap
        Colormap to convert
    n_colors : int, optional
        Number of colors in the colormap (default is 256).

    Returns
    -------
    List
        Plotly colorscale
    """
    colors = [colormap(i / (n_colors - 1)) for i in range(n_colors)]
    colorscale = [
        (i / (n_colors - 1), f"rgb({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})")
        for i, c in enumerate(colors)
    ]
    return colorscale


def reduce_points_number(df_input, n_cycle):
    """Reduces the number of points in the dataframe

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data to plot
    n_cycle : int
        Number of points that we want to keep for a single cycle

    Returns
    -------
    pandas.DataFrame
        DataFrame with reduced number of points
    """
    N = max(int(len(df_input) / (n_cycle * len(df_input["Cycle"].unique()))), 1)

    return df_input.iloc[::N]


def get_colorscale(state, nb_cycle, i):
    """Returns the Plotly colorscale for a given state

    Parameters
    ----------
    state : ['C', 'D']
        State of the cell charging
    nb_cycle : int
        Total number of cycles
    i : int
        Cycle number

    Returns
    -------
    Plotly colorscale
        Plotly colorscale corresponding to the charging state

    """
    colorscale_dict = {"C": "Blues", "D": "Reds"}
    colorscale = colorscale_dict[state]
    color_value = (i + 1) / (nb_cycle + 2)
    cmap = plt.get_cmap(colorscale)
    rgba_color = cmap(color_value)
    hex_color = f"rgba({int(rgba_color[0]*255)}, {int(rgba_color[1]*255)}, {int(rgba_color[2]*255)}, {rgba_color[3]})"

    return hex_color


def plot_test_over_time(df_input, file_path, save, png, plot, test="CCCV", pulse=False):
    """Plots the Current, Voltage and Capacity over TestTime for a given test type

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data to plot
    file_path : str
        Path to the file to process
    test : str, optional
        Type of test to plot (default is 'CCCV')
    pulse : bool, optional
        Whether the test type contains current pulses (default is False)
    save : bool, optional
        Whether to save the plots as .html files(default is False).
    png : bool, optional
        Whether to display the plots as .png files (default is False).

    Returns
    -------
    go.Figure
        Plot as a Plotly figure
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    param_dict = {
        "Voltage": "Voltage  /  V",
        "Current": "Current  /  A",
        "Capacity": "Capacity  /  Ah",
    }

    if pulse:
        df = pd.concat(df_input.values())
        N = max(1, int(len(df) / 100000))
        df = df.iloc[::N]
        hovertext = (
            "Pulse: "
            + df["Pulse"].astype(str)
            + "<br>Cycle: "
            + df["Cycle"].astype(str)
            + "<br>State: "
            + df["State"].astype(str)
            + "<br>C_Rate (h): "
            + df["C_Rate"].astype(str)
        )
    else:
        df = df_input.copy()
        N = max(1, int(len(df) / 100000))
        df = df.iloc[::N]
        hovertext = (
            "Cycle: "
            + df["Cycle"].astype(str)
            + "<br>State: "
            + df["State"].astype(str)
            + "<br>C_Rate (h): "
            + df["C_Rate"].astype(str)
        )

    fig = go.Figure()
    for param in param_dict.keys():
        fig.add_trace(
            go.Scatter(
                x=df["TestTime"],
                y=df[param],
                mode="lines",
                line=dict(width=line_width),
                text=hovertext,
                name=param_dict[param],
            )
        )

    fig.update_layout(
        title={
            "text": f"<b>Voltage Current and Capacity for {test}</b><br>" f"File : {file_name}",
            "font": {"size": title_size},
        },
        xaxis_title="<b>TestTime  /  s</b>",
        yaxis_title="<b>Voltage, Current and Capacity</b>",
        xaxis={"title": {"font": {"size": axis_size}}, "tickfont": {"size": tickfont_size}},
        yaxis={"title": {"font": {"size": axis_size}}, "tickfont": {"size": tickfont_size}},
        legend={"font": {"size": legend_size}},
    )

    if save:
        result_folder = os.path.join(folder_path, file_name)
        os.makedirs(result_folder, exist_ok=True)
        file_path = os.path.join(result_folder, f"{test}_over_time.json")
        fig.write_json(file_path)
    if png:
        png_image = fig.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    if plot:
        fig.show()

    return fig


def plot_DQDV_result(df, file_path, save, png, plot):
    """Plots the dQ/dV curve

    It gives the derivative of Capacity over Voltage

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data to plot
    file_path : str
        Path to the file to processn
    save : bool, optional
        Whether to save the plots as .html files(default is False).
    png : bool, optional
        Whether to display the plots as .png files (default is False).

    Returns
    -------
    go.Figure
        Plot as a Plotly figure
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    fig_dqdv = go.Figure()
    nb_cycle = df["Cycle"].max()
    for state in ["C", "D"]:
        for i, cycle in enumerate(df["Cycle"].unique()):
            df_state = df[(df["State"] == state) & (df["Cycle"] == cycle)].sort_values(by="smoothed_voltage")
            if len(df_state) > 0:

                hex_color = get_colorscale(state, nb_cycle, i)
                fig_dqdv.add_trace(
                    go.Scatter(
                        x=df_state["smoothed_voltage"],
                        y=df_state["smoothed_dqdv"],
                        mode="lines+markers",
                        marker=dict(
                            color=hex_color,
                            size=marker_size,
                        ),
                        line=dict(width=line_width),
                        hovertext=f"Cycle: {cycle}<br>" + "State: " + df_state["State"].astype(str),
                        name=f"Cycle {cycle}, State {state}",
                    )
                )

    fig_dqdv.update_layout(
        title={
            "text": f"<b>dQ/dV curves over cycles</b><br>",
            # f'<i>File : {file_name}</i>',
            "font": {"size": title_size},
        },
        xaxis_title="<b>Voltage  /  V</b>",
        yaxis_title="<b>dQ/dV  /  Ah/V</b>",
        xaxis={"title": {"font": {"size": axis_size}}, "tickfont": {"size": tickfont_size}},
        yaxis={"title": {"font": {"size": axis_size}}, "tickfont": {"size": tickfont_size}},
        legend={"font": {"size": legend_size}},
    )

    if save:
        result_folder = os.path.join(folder_path, file_name)
        os.makedirs(result_folder, exist_ok=True)
        file_path_dqdv = os.path.join(result_folder, f"dQdV.json")
        fig_dqdv.write_json(file_path_dqdv)
    if png:
        png_image = fig_dqdv.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    if plot:
        fig_dqdv.show()

    return fig_dqdv


def plot_dqdv_heatmap(df, file_path, save, png, plot):
    """Plots the dQ/dV heatmap

    It gives a plot with Cycles over Voltage colored by the magnitude of dQ/dV

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data to plot
    file_path : str
        Path to the file to process
    save : bool, optional
        Whether to save the plots as .html files(default is False).
    png : bool, optional
        Whether to display the plots as .png files (default is False).

    Returns
    -------
    go.Figure
        Plot as a Plotly figure
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    batlow_colorscale = matplotlib_to_plotly_colorscale(cm.lipari)

    fig_dqdv = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=["Charge", "Discharge"]
    )

    for i_charge, charging_state in enumerate(["C", "D"]):
        df_charge = df[df["State"] == charging_state]

        if charging_state == "C":
            colorbar_title = "log|dQ/dV| for <b>charge</b>"
            yanchor = "bottom"
        else:
            colorbar_title = "log|dQ/dV| for <b>discharge</b>"
            yanchor = "top"

        fig_dqdv.add_trace(
            go.Heatmap(
                x=df_charge["smoothed_voltage"],
                y=df_charge["Cycle"],
                z=np.log(abs(df_charge["smoothed_dqdv"])),
                colorscale=batlow_colorscale,
                # colorbar_title=colorbar_title,
                colorbar=dict(
                    title=dict(text=colorbar_title, font=dict(size=legend_size)),
                    yanchor=yanchor,
                    len=0.5,
                ),
            ),
            row=i_charge + 1,
            col=1,
        )
        fig_dqdv.update_xaxes(title="Voltage (V)", title_font=dict(size=axis_size), row=i_charge + 1, col=1)
        fig_dqdv.update_yaxes(title="Cycle", title_font=dict(size=axis_size), row=i_charge + 1, col=1)

    fig_dqdv.update_layout(
        title="dQ/dV heatmap showing its magnitude's evolution through cycles<br>",
        # f'File : {file_name}',
        title_font=dict(size=title_size),
        font=dict(size=20),
    )

    if save:
        result_folder = os.path.join(folder_path, file_name)
        os.makedirs(result_folder, exist_ok=True)
        file_path_heatmap = os.path.join(result_folder, f"dqdv_heatmap.json")
        fig_dqdv.write_json(file_path_heatmap)
    if png:
        png_image = fig_dqdv.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    if plot:
        fig_dqdv.show()

    return fig_dqdv


def plot_pocv(df, file_path, save, png, plot):
    """Plots the POCV curve

    It gives a plot of the Capacity over Voltage

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data to plot
    file_path : str
        Path to the file to process
    save : bool, optional
        Whether to save the plots as .html files(default is False).
    png : bool, optional
        Whether to display the plots as .png files (default is False).

    Returns
    -------
    go.Figure
        Plot as a Plotly figure
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    df = reduce_points_number(df, n_cycle=1000)

    fig_pocv = go.Figure()
    nb_cycle = len(df["Cycle"].unique())

    for state in ["C", "D"]:
        for i, cycle in enumerate(df["Cycle"].unique()):
            df_state = df[(df["State"] == state) & (df["Cycle"] == cycle) & (df["normcurrent"] != 0)].sort_values(
                by="Voltage"
            )
            if len(df_state) > 0:

                hex_color = get_colorscale(state, nb_cycle, i)
                fig_pocv.add_trace(
                    go.Scatter(
                        x=df_state["Capacity"],
                        y=df_state["Voltage"],
                        mode="lines+markers",
                        marker=dict(
                            color=hex_color,
                            size=marker_size,
                        ),
                        line=dict(width=3),
                        hovertext=f"Cycle: {cycle}<br>"
                        + "State: "
                        + df_state["State"].astype(str)
                        + "C_Rate (h): "
                        + df_state["C_Rate"].astype(str),
                        name=f"Cycle {cycle}, State {state}",
                    )
                )

    fig_pocv.update_layout(
        title={
            "text": f"<b>POCV curve over cycles</b><br>",
            # f'File : {file_name}',
            "font": {"size": title_size},
        },
        xaxis_title="<b>Capacity  /  Ah</b>",
        yaxis_title="<b>Voltage  /  V</b>",
        xaxis={"title": {"font": {"size": axis_size}}, "tickfont": {"size": tickfont_size}},
        yaxis={"title": {"font": {"size": axis_size}}, "tickfont": {"size": tickfont_size}},
        legend={"font": {"size": legend_size}},
    )

    if save:
        result_folder = os.path.join(folder_path, file_name)
        os.makedirs(result_folder, exist_ok=True)
        file_path_pocv = os.path.join(result_folder, f"pocv.json")
        fig_pocv.write_json(file_path_pocv)
    if png:
        png_image = fig_pocv.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    if plot:
        fig_pocv.show()

    return fig_pocv


def plot_GITT_result(df, file_path, save, png, plot, column="Diffusion Coefficient", test="GITT"):
    """Plots the GITT results curve

    It gives a plot of the selected parameter over SOC

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data to plot
    file_path : str
        Path to the file to process
    column : str, optional
        Column to plot (default is 'Diffusion Coefficient').
    save : bool, optional
        Whether to save the plots as .html files(default is False).
    png : bool, optional
        Whether to display the plots as .png files (default is False).

    Returns
    -------
    go.Figure
        Plot as a Plotly figure
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    fig_GITT = go.Figure()
    nb_cycle = len(df["Cycle"].unique())
    for state in ["C", "D"]:
        for i, cycle in enumerate(df["Cycle"].unique()):
            df_state = df[(df["State"] == state) & (df["Cycle"] == cycle)].sort_values(by="SOC")

            hex_color = get_colorscale(state, nb_cycle, i)

            fig_GITT.add_trace(
                go.Scatter(
                    x=df_state["SOC"],
                    y=df_state[column],
                    mode="lines+markers",
                    marker=dict(color=hex_color, size=marker_size),
                    line=dict(width=line_width),
                    hovertext=f"Cycle: {cycle}<br>" + "Pulse: " + df_state["Pulse"].astype(str),
                    name=f"Cycle {cycle}, State {state}",
                )
            )

    fig_GITT.update_layout(
        title={
            "text": f"<b>Parameter numerical results for {test} Test</b><br>" f"File : {file_name}",
            "font": {"size": title_size},
        },
        xaxis_title="SOC",
        yaxis_title=column,
        xaxis={"title": {"font": {"size": axis_size}}, "tickfont": {"size": tickfont_size}},
        yaxis={"title": {"font": {"size": axis_size}}, "tickfont": {"size": tickfont_size}},
        legend={"font": {"size": legend_size}},
    )

    if save:
        result_folder = os.path.join(folder_path, file_name)
        os.makedirs(result_folder, exist_ok=True)
        file_path_GITT = os.path.join(result_folder, f"GITT_{column}.json")
        fig_GITT.write_json(file_path_GITT)
    if png:
        png_image = fig_GITT.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    if plot:
        fig_GITT.show()

    return fig_GITT


def plot_HPPC_result(df, file_path, save, png, plot, column="R"):
    """Plots the HPPC results curve

    It gives a plot of the selected parameter over SOC for both charge and dicharge pulses

    Parameters
    ----------
    df_input : pandas.DataFrame
        DataFrame containing the data to plot
    file_path : str
        Path to the file to process
    save : bool, optional
        Whether to save the plots as .html files(default is False).
    png : bool, optional
        Whether to display the plots as .png files (default is False).

    Returns
    -------
    go.Figure
        Plot as a Plotly figure
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.dirname(file_path)

    fig_HPPC = go.Figure()
    nb_cycle = df["Cycle"].max()
    for state in ["C", "D"]:
        for i, cycle in enumerate(df["Cycle"].unique()):
            df_state = df[(df["State"] == state) & (df["Cycle"] == cycle)].sort_values(by="SOC")

            for charge in ["charge", "discharge"]:
                hex_color = get_colorscale("C", nb_cycle, i) if charge == "charge" else get_colorscale("D", nb_cycle, i)

                fig_HPPC.add_trace(
                    go.Scatter(
                        x=df_state["SOC"],
                        y=df_state[f"{column}_{charge}"],
                        mode="lines+markers",
                        marker=dict(
                            color=hex_color,
                            size=marker_size,
                        ),
                        line=dict(width=line_width),
                        hovertext=f"Cycle: {cycle}<br>" + "Pulse: " + df_state["Pulse"].astype(str),
                        name=f"Cycle {cycle}, {charge} pulse",
                    )
                )

    fig_HPPC.update_layout(
        title={
            "text": f"<b>Parameter numerical results for HPPC Test</b><br>" f"File : {file_name}",
            "font": {"size": title_size},
        },
        xaxis_title="SOC",
        yaxis_title=column,
        xaxis={"title": {"font": {"size": axis_size}}, "tickfont": {"size": tickfont_size}},
        yaxis={"title": {"font": {"size": axis_size}}, "tickfont": {"size": tickfont_size}},
        legend={"font": {"size": legend_size}},
    )

    if save:
        result_folder = os.path.join(folder_path, file_name)
        os.makedirs(result_folder, exist_ok=True)
        file_path_HPPC = os.path.join(result_folder, f"HPPC_{column}.json")
        fig_HPPC.write_json(file_path_HPPC)
    if png:
        png_image = fig_HPPC.to_image(format="png", width=2000, height=800)
        display(Image(data=png_image))
    if plot:
        fig_HPPC.show()

    return fig_HPPC
