import pandas as pd
import numpy as np
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

    df['Pulse'] = (
        (df['normcurrent'] != 0).astype(int).diff().fillna(0).gt(0).cumsum().ffill().shift(-1)
    )
    df['Relaxation'] = (df['normcurrent'] == 0).astype(int)
    df['not_pulse'] = 0

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
    df_nested = {int(pulse): df[df['Pulse'] == pulse] for pulse in df['Pulse'].unique()}

    return df_nested


def global_calculation_GITT(df_nested, my_func_list):
    """Calculates the GITT parameters

    Given a nested DataFrame containing the GITT data, this function calculates the GITT parameters for each pulse

    Parameters
    ----------
    df_nested : Dict
        Dict containing the nested DataFrames corresponding to each pulse

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the GITT parameters structured by pulse number
    """
    df_total = pd.DataFrame()
    for pulse, df_pulse in df_nested.items():
        enough_relaxation_data = len(df_pulse[df_pulse['Relaxation'] == 1]) > 2
        enough_pulse_data = len(df_pulse[df_pulse['Relaxation'] == 0]) > 2
        deltaV = abs(df_pulse['Voltage'].iloc[0] - df_pulse['Voltage'].iloc[-1])
        is_a_pulse = deltaV > 0.0001
        if enough_relaxation_data and enough_pulse_data and is_a_pulse:

            try:
                SOC = df_pulse['SOC'].iloc[0]
                if SOC < -0.01 or SOC > 1.01:
                    continue
                V0, V1, V2, V3, t0, t1, t2, t3, Ipulse, delta_Eocv, delta_Edrop, delta_Epulse = (
                    calculate_relevant_points_GITT(df_pulse)
                )
                Voltage = interp1d(df_pulse['TestTime'], df_pulse['Voltage'], kind='linear')
                Current = interp1d(df_pulse['TestTime'], df_pulse['Current'], kind='linear')
                Capacity = interp1d(df_pulse['TestTime'], df_pulse['Capacity'], kind='linear')

                tau = t2 - t1
                resistance = abs((V1 - V0) / Ipulse)

                rc_est = estimate_2rc_from_relaxation(df_pulse, Ipulse, t1, t2)

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
                        'tau1_s': rc_est['tau1'],
                        'R1_ohm': rc_est['R1'],
                        'C1_F': rc_est['C1'],
                        'tau2_s': rc_est['tau2'],
                        'R2_ohm': rc_est['R2'],
                        'C2_F': rc_est['C2'],
                    },
                    index=['Pulse'],
                )

                # Parameters that can be used in an external function
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
                    'State': df_pulse['State'].iloc[0],
                }

                for func in my_func_list:
                    var_name = func.__name__
                    new_var = add_function(func, **kwargs)
                    if new_var is not None:
                        df_coefficient[var_name] = new_var

                df_total = pd.concat([df_total, df_coefficient])

            except Exception as e:
                print("error: ", e)

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


def estimate_2rc_from_relaxation(df_pulse, Ipulse, t1, t2, min_points=6):
    """
    Estimate two RC branches from the post-pulse relaxation.

    Parameters
    ----------
    df_pulse : pd.DataFrame
        The nested dataframe (one pulse) as in your pipeline.
    Ipulse : float
        Pulse current (use absolute value; must be in A).
    t1 : float
        Pulse start time (TestTime)
    t2 : float
        Pulse end time (TestTime)
    min_points : int
        Minimum number of relaxation points required for a fit.

    Returns
    -------
    dict
        {
          'Vinf': float,
          'A1': float, 'tau1': float, 'R1': float, 'C1': float,
          'A2': float, 'tau2': float, 'R2': float, 'C2': float,
          'success': bool
        }
        NaNs returned for values if fit failed or physically invalid.
    """
    # Prepare results default
    res = dict.fromkeys(
        ['Vinf', 'A1', 'tau1', 'R1', 'C1', 'A2', 'tau2', 'R2', 'C2', 'success'], np.nan
    )
    res['success'] = False

    # Safety
    if Ipulse is None or not np.isfinite(Ipulse) or abs(Ipulse) < 1e-12:
        return res

    # pull relaxation after pulse
    df_relax_post = df_pulse[df_pulse['Relaxation'] == 1].copy()
    # relative time from pulse end
    df_relax_post['t_rel'] = df_relax_post['TestTime'] - t2

    if len(df_relax_post) < min_points:
        return res

    t = df_relax_post['t_rel'].values
    V = df_relax_post['Voltage'].values

    # initial guesses:
    Vinf_guess = V[-1]
    A_guess_total = V[0] - V[-1]
    # split amplitude guesses
    A1_guess = 0.6 * A_guess_total
    A2_guess = 0.4 * A_guess_total
    # taus: rough guesses based on timespan
    tspan = max(t.max() - t.min(), 1e-6)
    tau1_guess = max(0.1, 0.05 * tspan)  # short
    tau2_guess = max(1.0, 0.5 * tspan)  # longer

    p0 = [Vinf_guess, A1_guess, tau1_guess, A2_guess, tau2_guess]

    # bounds: Vinf free, A's can be pos/neg but limited, taus positive
    # tau bounds from 1e-3 s to e.g. 1e5 s (very wide)
    lower = [-np.inf, -np.inf, 1e-4, -np.inf, 1e-4]
    upper = [np.inf, np.inf, 1e6, np.inf, 1e6]

    try:
        popt, pcov = curve_fit(_two_exp, t, V, p0=p0, bounds=(lower, upper), maxfev=20000)
    except Exception:
        return res

    Vinf, A1, tau1, A2, tau2 = popt
    # reorder so tau1 < tau2 (canonical)
    if tau1 > tau2:
        # swap components
        A1, A2 = A2, A1
        tau1, tau2 = tau2, tau1

    # sanity checks on fitted taus
    if tau1 <= 0 or tau2 <= 0 or not np.isfinite(tau1) or not np.isfinite(tau2):
        return res

    # pulse duration
    delta_t = t2 - t1
    # avoid divide-by-zero or extremely tiny denominators for extremely short pulses
    Ip = abs(Ipulse)

    # Convert amplitude to R_i using finite-duration correction:
    def _Ri_from_Ai(Ai, tau_i):
        denom = Ip * (1.0 - np.exp(-delta_t / tau_i))
        if abs(denom) < 1e-12:
            return np.nan
        return -Ai / denom

    R1 = _Ri_from_Ai(A1, tau1)
    R2 = _Ri_from_Ai(A2, tau2)

    # compute C = tau / R (if R valid and nonzero)
    C1 = np.nan
    C2 = np.nan
    if np.isfinite(R1) and abs(R1) > 0:
        C1 = tau1 / R1
    if np.isfinite(R2) and abs(R2) > 0:
        C2 = tau2 / R2

    # fill results
    res.update(
        {
            'Vinf': float(Vinf),
            'A1': float(A1),
            'tau1': float(tau1),
            'R1': float(R1),
            'C1': float(C1),
            'A2': float(A2),
            'tau2': float(tau2),
            'R2': float(R2),
            'C2': float(C2),
            'success': True,
        }
    )

    # Basic physical sanity checks: resistances positive and times sensible
    if (not np.isfinite(R1) or R1 <= 0) and (not np.isfinite(R2) or R2 <= 0):
        # both invalid -> mark fail
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
