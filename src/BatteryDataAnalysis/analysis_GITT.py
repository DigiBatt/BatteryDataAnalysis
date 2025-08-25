import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from .create_custom_functions import add_function
from whittaker_eilers import WhittakerSmoother
from scipy.interpolate import interp1d


def pulse_number_GITT(df_input):
    """Pulse selection for GITT data

    - For every cycles, gives a pulse number for every current pulses.
    - Then it gives this number for the last point before the pulse until the last point before the end of the relaxation.
    - Finally, it creates a nested Dataframe structuring the GITT data for each pulse.

    Here is a schema of a selected pulse and its relevant points:

    .. image:: /_static/pulse_GITT.png
        :width: 2000px
        :align: center
        :alt: Example of a pulse selected by the function for GITT data


    The corresponding parameters that can be used in a custom function (see :func:`add_function <BatteryDataAnalysis.create_custom_functions.add_function>`) are:
    *V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Eocv, delta_Edrop, delta_Epulse* that are calculated in :func:`calculate_relevant_points_GITT`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the GITT data to process

    Returns
    -------
    dict
        Dict containing the nested DataFrames as values and their pulse number as keys
    """
    # A pulse starts the time just before the pulse and ends the time just before the next pulse at the end of the relaxation time
    df = df_input.copy()

    # Pulse labeling: each time current goes from 0 -> nonzero, increment ID
    df['Pulse'] = (
        (df['normcurrent'] != 0).astype(int).diff().fillna(0).gt(0).cumsum().ffill().shift(-1)
    )
    df['Relaxation'] = (df['normcurrent'] == 0).astype(int)
    df['not_pulse'] = 0

    # Clean up spurious pulses as before
    for cycle in df['Cycle'].unique():
        for state in df[df['Cycle'] == cycle]['State'].unique():
            df_state = df[(df['State'] == state) & (df['Cycle'] == cycle)].copy()
            df_group = (
                df_state[df_state['normcurrent'] != 0]
                .groupby('Pulse')['TestTime']
                .agg(lambda x: x.max() - x.min())
            )
            pulse_time = df_group.mean()
            for pulse in df_state['Pulse'].unique():
                df_pulse = df_state[(df_state['Pulse'] == pulse) & (df_state['normcurrent'] != 0)]
                if df_pulse['TestTime'].max() - df_pulse['TestTime'].min() > pulse_time * 10:
                    df.loc[
                        (df['State'] == state) & (df['Cycle'] == cycle) & (df['Pulse'] == pulse),
                        'not_pulse',
                    ] = 1

    df = df.dropna(subset=['Pulse'])

    # Create dicts
    df_nested_pulse = {int(p): df[(df['Pulse'] == p)] for p in df['Pulse'].unique()}
    df_nested_relax = {
        int(p): df[(df['Pulse'] == p) & (df['normcurrent'] == 0)] for p in df['Pulse'].unique()
    }

    return df_nested_pulse, df_nested_relax


def global_calculation_GITT(df_nested_pulse, df_nested_relax, my_func_list):
    """
    Calculates the GITT parameters from pulse + relaxation data

    Parameters
    ----------
    df_nested_pulse : dict
        Dict containing pulse-only DataFrames
    df_nested_relax : dict
        Dict containing relaxation-only DataFrames
    my_func_list : list
        List of user-defined functions to apply

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the GITT parameters structured by pulse number
    """
    df_total = pd.DataFrame()

    # iterate only over pulse dict keys
    for pulse, df_pulse in df_nested_pulse.items():
        if pulse not in df_nested_relax:
            continue  # skip if no relaxation segment found

        df_relax = df_nested_relax[pulse]

        # data availability checks
        enough_relaxation_data = len(df_relax) > 2
        enough_pulse_data = len(df_pulse) > 2
        deltaV = abs(df_pulse['Voltage'].iloc[0] - df_pulse['Voltage'].iloc[-1])
        is_a_pulse = deltaV > 0.0001

        if enough_relaxation_data and enough_pulse_data and is_a_pulse:
            try:
                SOC = df_pulse['SOC'].iloc[0]
                if SOC < -0.01 or SOC > 1.01:
                    continue

                # Calculate relevant points on the pulse
                V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Eocv, delta_Edrop, delta_Epulse = (
                    calculate_relevant_points_GITT(df_pulse)
                )

                # interpolators
                Voltage = interp1d(df_pulse['TestTime'], df_pulse['Voltage'], kind='linear')
                Current = interp1d(df_pulse['TestTime'], df_pulse['Current'], kind='linear')
                Capacity = interp1d(df_pulse['TestTime'], df_pulse['Capacity'], kind='linear')

                tau = t2 - t1
                resistance = abs((V1 - V0) / Ipulse)

                # Use relaxation segment for RC estimation
                rc_est = estimate_2rc_from_relaxation_split(df_relax, Ipulse, t1, t2)

                D = calculate_diffusion_coefficient_GITT(delta_Eocv, delta_Epulse, tau)

                R_1s = calculate_DCIR(V0, Voltage, t0, t3, Ipulse, 1)
                R_30s = calculate_DCIR(V0, Voltage, t0, t3, Ipulse, 30)
                R_60s = calculate_DCIR(V0, Voltage, t0, t3, Ipulse, 60)
                R_180s = calculate_DCIR(V0, Voltage, t0, t3, Ipulse, 178)

                df_coefficient = pd.DataFrame(
                    {
                        'Pulse': pulse,
                        'Cycle': df_pulse['Cycle'].iloc[-1],
                        'State': df_pulse['State'].iloc[-1],
                        'TestTime': t0,
                        'SOC': SOC,
                        'OCV': V3,
                        'Ipulse': Ipulse,
                        'Diffusion Coefficient': D,
                        'Ohmic Resistance': resistance,
                        'tau1': rc_est['tau1'],
                        'R1': rc_est['R1'],
                        'C1': rc_est['C1'],
                        'tau2': rc_est['tau2'],
                        'R2': rc_est['R2'],
                        'C2': rc_est['C2'],
                        'R_1s': R_1s,
                        'R_30s': R_30s,
                        'R_60s': R_60s,
                        'R_180s': R_180s,
                    },
                    index=[pulse],
                )

                # Apply custom functions
                kwargs = {
                    'V0': V0,
                    'V1': V1,
                    'V2': V2,
                    'V3': V3,
                    't0': t0,
                    't1': t1,
                    't2': t2,
                    't3': t3,
                    'Ipulse': Ipulse,
                    'delta_Eocv': delta_Eocv,
                    'delta_Edrop': delta_Edrop,
                    'delta_Epulse': delta_Epulse,
                    'Voltage': Voltage,
                    'Current': Current,
                    'Capacity': Capacity,
                    'df_pulse': df_pulse,
                    'df_relax': df_relax,
                    'State': df_pulse['State'].iloc[0],
                }

                for func in my_func_list:
                    var_name = func.__name__
                    new_var = add_function(func, **kwargs)
                    if new_var is not None:
                        df_coefficient[var_name] = new_var

                df_total = pd.concat([df_total, df_coefficient])

            except Exception as e:
                print(f"error in pulse {pulse}: {e}")

    return df_total


def calculate_relevant_points_GITT(df_pulse):
    """Calculates the relevant points represented in the :func:`pulse_number_GITT` documentation.

    These parameters can be used in a custom function to calculate new parameters (see :func:`add_function <BatteryDataAnalysis.create_custom_functions.add_function>`).

    Parameters
    ----------
    df_pulse : pandas.DataFrame
        DataFrame containing the data corresponding to one pulse

    Returns
    -------
    Tuple
        Tuple containing the relevant points : V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Eocv, delta_Edrop, delta_Epulse
    """
    # Split pulse vs relaxation correctly
    df_load = df_pulse[df_pulse['Relaxation'] == 0].copy()  # under current
    df_relax = df_pulse[df_pulse['Relaxation'] == 1].copy()  # relaxation

    # print(df_pulse[['TestTime', 'normcurrent', 'Relaxation']].head(20))

    # Identify start/end of the pulse in time
    t_pulse_start = df_load['TestTime'].iloc[0]
    t_pulse_end = df_load['TestTime'].iloc[-1]

    # Pre- and post- relaxation around the pulse
    df_relax_pre = df_relax[df_relax['TestTime'] < t_pulse_start].copy()
    df_relax_post = df_relax[df_relax['TestTime'] > t_pulse_end].copy()

    # Smooth ONLY the post-relaxation segment (the steady-state you want as OCV_after)
    if len(df_relax_post) >= 3:
        signal_noise = np.diff(df_relax_post['Voltage'], 2)
        signal_noise_absolute = np.abs(signal_noise).reshape(-1, 1)
        signal_noise_normalized = signal_noise_absolute / abs(df_relax_post['Voltage']).max()
        weights = np.exp(-signal_noise_normalized)
        weights_padded = np.concatenate(([weights[0]], weights, [weights[-1]]))
        whittaker_smoother = WhittakerSmoother(
            lmbda=1e3,
            order=1,
            data_length=len(df_relax_post),
            x_input=df_relax_post['TestTime'],
            weights=weights_padded,
        )
        df_relax_post['Voltage'] = whittaker_smoother.smooth(df_relax_post['Voltage'].values)

    # Relevant points (conventional GITT definitions)
    # V0: last OCV before the pulse (end of pre-relaxation)
    V0 = df_relax_pre['Voltage'].iloc[-1] if len(df_relax_pre) else df_pulse['Voltage'].iloc[0]
    # V1: voltage immediately after current step (start of pulse, includes IR drop)
    V1 = df_load['Voltage'].iloc[0]
    # V2: end of pulse (just before relaxation begins)
    V2 = df_load['Voltage'].iloc[-1]
    # V3: OCV after full relaxation (end of post-relaxation)
    V3 = df_relax_post['Voltage'].iloc[-1] if len(df_relax_post) else df_pulse['Voltage'].iloc[-1]

    t0 = df_relax_pre['TestTime'].iloc[-1] if len(df_relax_pre) else df_pulse['TestTime'].iloc[0]
    t1 = df_load['TestTime'].iloc[0]
    t2 = df_load['TestTime'].iloc[-1]
    t3 = df_relax_post['TestTime'].iloc[-1] if len(df_relax_post) else df_pulse['TestTime'].iloc[-1]

    Ipulse = df_pulse[df_pulse['normcurrent'] != 0]['Current'].mean()

    # OCV and drops (typical GITT)
    delta_Eocv = V3 - V0  # steady-state change
    delta_Edrop = V1 - V0  # instantaneous IR drop
    delta_Epulse = V2 - V1  # diffusion-related during pulse

    return V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Eocv, delta_Edrop, delta_Epulse


def _two_exp(t, Vinf, A1, tau1, A2, tau2):
    """Model: V(t) = Vinf + A1*exp(-t/tau1) + A2*exp(-t/tau2)."""
    return Vinf + A1 * np.exp(-t / np.abs(tau1)) + A2 * np.exp(-t / np.abs(tau2))


def _single_exp(t, Vinf, A, tau):
    return Vinf + A * np.exp(-t / tau)


def estimate_2rc_from_relaxation(
    df_relax, Ipulse, t1, t2, short_window=100, long_multiplier=5, smooth_window=11, polyorder=2
):
    """
    Estimate two RC branches from post-pulse relaxation, following the methodology from the IEEE paper:
    - First RC branch: fit short transient (~short_window seconds)
    - Second RC branch: fit remaining long transient after subtracting first RC contribution
    - R and C calculated from steady-state voltages

    Parameters
    ----------
    df_relax : pd.DataFrame
        Post-pulse relaxation dataframe with 'TestTime' and 'Voltage'
    Ipulse : float
        Pulse current (A)
    t1 : float
        Pulse start time
    t2 : float
        Pulse end time
    short_window : float
        Duration of short transient (s)
    long_multiplier : float
        Multiplier for long transient duration relative to short_window
    smooth_window : int
        Savitzky-Golay filter window length
    polyorder : int
        Savitzky-Golay filter polynomial order

    Returns
    -------
    dict
        {'Vinf', 'R1', 'C1', 'R2', 'C2', 'success'}
    """
    res = dict.fromkeys(['Vinf', 'R1', 'C1', 'R2', 'C2', 'success'], np.nan)
    res['success'] = False

    if Ipulse is None or not np.isfinite(Ipulse) or abs(Ipulse) < 1e-12:
        return res

    df_post = df_relax[df_relax['Relaxation'] == 1].copy()
    df_post['t_rel'] = df_post['TestTime'] - t2
    df_post = df_post[df_post['t_rel'] >= 0]

    if len(df_post) < 5:
        return res

    t = df_post['t_rel'].values
    V = df_post['Voltage'].values

    # smooth voltage
    if len(V) >= smooth_window:
        V = savgol_filter(V, window_length=smooth_window, polyorder=polyorder)

    delta_t = max(t2 - t1, 1e-6)
    Ip_abs = abs(Ipulse)

    # Short transient (first RC branch)
    short_idx = t <= short_window
    if not np.any(short_idx):
        return res

    V_short = V[short_idx]
    t_short = t[short_idx]

    V0 = V_short[0]
    Vinf_guess = V_short[-1]
    A1 = V0 - Vinf_guess
    tau1 = short_window / 2  # rough guess
    # Steady-state voltage for first RC branch
    Vss1 = V_short[-1]
    R1 = max(Vss1 / Ip_abs, 0)  # enforce positive resistance
    C1 = tau1 / R1 if R1 > 0 else np.nan

    # Subtract first RC contribution to isolate long transient
    V_RC1 = A1 * np.exp(-t / tau1)
    V_residual = V - V_RC1 - Vinf_guess

    # Long transient (second RC branch)
    long_duration = long_multiplier * short_window
    long_idx = (t > short_window) & (t <= short_window + long_duration)
    if np.any(long_idx):
        V_long = V_residual[long_idx]
        t_long = t[long_idx]
        if len(V_long) >= 3:
            A2 = V_long[0]  # initial amplitude
            tau2 = long_duration / 2
            Vss2 = V_long[-1]
            R2 = max(Vss2 / Ip_abs, 0)
            C2 = tau2 / R2 if R2 > 0 else np.nan
        else:
            R2 = C2 = np.nan
    else:
        R2 = C2 = np.nan

    res.update({'Vinf': Vinf_guess, 'R1': R1, 'C1': C1, 'R2': R2, 'C2': C2, 'success': True})
    return res


def estimate_2rc_from_relaxation_split(
    df_relax, Ipulse, t1, t2, min_points=6, log_subsample=True, smooth_window=11, polyorder=2
):
    """
    Estimate one or two RC branches from the post-pulse relaxation, handling both
    charging and discharging pulses. Optionally applies logarithmic subsampling and smoothing.

    Parameters
    ----------
    df_relax : pd.DataFrame
        Nested dataframe for one pulse.
    Ipulse : float
        Pulse current (can be positive or negative).
    t1 : float
        Pulse start time (TestTime)
    t2 : float
        Pulse end time (TestTime)
    min_points : int
        Minimum number of relaxation points required.
    log_subsample : bool
        Whether to logarithmically subsample t_rel to stabilize fitting.
    smooth_window : int
        Window size for Savitzky-Golay smoothing (must be odd).
    polyorder : int
        Polynomial order for Savitzky-Golay filter.

    Returns
    -------
    dict
        {'Vinf','A1','tau1','R1','C1','A2','tau2','R2','C2','success'}
    """
    res = dict.fromkeys(
        ['Vinf', 'A1', 'tau1', 'R1', 'C1', 'A2', 'tau2', 'R2', 'C2', 'success'], np.nan
    )
    res['success'] = False

    if Ipulse is None or not np.isfinite(Ipulse) or abs(Ipulse) < 1e-12:
        return res

    # Only post-pulse relaxation
    df_post = df_relax[df_relax['Relaxation'] == 1].copy()
    df_post['t_rel'] = df_post['TestTime'] - t2
    df_post = df_post[df_post['t_rel'] >= 0]

    if len(df_post) < min_points:
        return res

    t = df_post['t_rel'].values
    V = df_post['Voltage'].values

    # Smooth voltage
    if len(V) >= smooth_window:
        V = savgol_filter(V, window_length=smooth_window, polyorder=polyorder)

    delta_t = max(t2 - t1, 1e-6)
    Ip_abs = abs(Ipulse)  # use absolute current for resistance calculation

    # Logarithmic subsampling
    if log_subsample and len(t) > 1000:
        n_points = 200
        t_log = np.logspace(np.log10(t[0] + 1e-12), np.log10(t[-1] + 1e-12), n_points)
        V_log = np.interp(t_log, t, V)
        t, V = t_log, V_log

    # Initial guesses
    Vinf_guess = V[-1]
    A_total = V[0] - V[-1]

    # Small voltage change -> single-exp approximation
    if abs(A_total) < 1e-4:
        return {
            'Vinf': V[-1],
            'A1': 0.0,
            'tau1': 1.0,
            'R1': np.nan,
            'C1': np.nan,
            'A2': 0.0,
            'tau2': np.nan,
            'R2': np.nan,
            'C2': np.nan,
            'success': True,
        }

    # Two-exp initial guesses
    A1_guess = 0.6 * A_total
    A2_guess = 0.4 * A_total
    tspan = max(t.max() - t.min(), 1e-6)
    tau1_guess = max(0.1, 0.05 * tspan)
    tau2_guess = max(1.0, 0.5 * tspan)
    p0 = [Vinf_guess, A1_guess, tau1_guess, A2_guess, tau2_guess]
    lower = [-np.inf, -np.inf, 1e-4, -np.inf, 1e-4]
    upper = [np.inf, np.inf, 1e6, np.inf, 1e6]

    # Fit two-exponential
    try:
        popt, _ = curve_fit(_two_exp, t, V, p0=p0, bounds=(lower, upper), maxfev=20000)
        Vinf, A1, tau1, A2, tau2 = popt
        if tau1 > tau2:
            A1, A2 = A2, A1
            tau1, tau2 = tau2, tau1
    except Exception:
        # fallback: single-exponential
        try:
            popt, _ = curve_fit(
                _single_exp, t, V, p0=[Vinf_guess, A_total, tspan / 2], maxfev=10000
            )
            Vinf, A1, tau1 = popt
            A2, tau2 = 0.0, np.nan
        except Exception:
            return res

    # Convert amplitudes to resistances using absolute pulse current
    def _Ri(Ai, tau_i, Ipulse, delta_t):
        denom = Ipulse * (1 - np.exp(-delta_t / tau_i))
        if abs(denom) < 1e-12:
            return np.nan
        return Ai / denom

    R1 = _Ri(A1, tau1, Ipulse, delta_t)
    R2 = _Ri(A2, tau2, Ipulse, delta_t) if A2 != 0 else np.nan
    C1 = tau1 / R1 if np.isfinite(R1) and abs(R1) > 0 else np.nan
    C2 = tau2 / R2 if np.isfinite(R2) and abs(R2) > 0 else np.nan

    # Fill results
    res.update(
        {
            'Vinf': Vinf,
            'A1': A1,
            'tau1': tau1,
            'R1': R1,
            'C1': C1,
            'A2': A2,
            'tau2': tau2,
            'R2': R2,
            'C2': C2,
            'success': True,
        }
    )

    # sanity check: at least one positive R
    if (not np.isfinite(R1) or R1 <= 0) and (not np.isfinite(R2) or R2 <= 0):
        res['success'] = False

    return res


def calculate_DCIR(V0, Voltage, t0, t3, Ipulse, r_time):
    """Calculates the DCIR for GITT test

    Parameters
    ----------
    V0 : float
    Voltage : interp1d
        Interpolation function of the voltage
    t0 : float
    t3 : float
    Ipulse : float
    r_time : float
        Time after which DCIR is calculated

    Returns
    -------
    Float
        DCIR of the pulse
    """
    if t0 + r_time <= t3:
        V_r_time = Voltage(t0 + r_time)
        resistance = abs((V_r_time - V0) / Ipulse)
    else:
        resistance = np.nan

    return resistance


def calculate_diffusion_coefficient_GITT(delta_Eocv, delta_Epulse, tau):
    r"""Calculates diffusion coefficient for GITT test

    See the :func:`pulse_number_GITT` documentation for the points definition.
    The formula comes from the paper from T. Schied et al. (2021)
    *"Determining the Diffusion Coefficient of Lithium Insertion Cathodes from GITT measurements: Theoretical Analysis for low Temperatures"*

    The diffusion coefficient is calculated using the following formula:

    .. math::

        D = \frac{4}{9 \pi} \cdot \frac{\text{radius}^2}{\tau} \cdot \left(\frac{\Delta E_{OCV}}{\Delta E_{pulse}}\right)^2

    Parameters
    ----------
    V0 : float
    V1 : float
    V2 : float
    V3 : float
    tau : float
        Time duration of the pulse

    Returns
    -------
    float
        Diffusion coefficient of the selected pulse
    """
    radius = 5e-6

    D = 4 / (9 * np.pi) * radius**2 / tau * (delta_Eocv / delta_Epulse) ** 2
    return D
