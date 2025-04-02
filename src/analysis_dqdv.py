import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


def calculate_dqdv_for_all_cycle(df, smoothing=True):
    """Calculate dQ/dV for a DataFrame for all the cycles with the columns 'Cycle', 'Voltage', 'Capacity' and 'Current'.
    
    Parameters
    -------
    df : pandas.DataFrame
        DataFrame containing the data
    smoothing : bool, optional
        Whether to smooth the dQ/dV curve

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the DataFrame with the dQ/dV curve and its performance metrics
    """
    df_dqdv = pd.DataFrame()

    for charging_state in ['D', 'C']:
        sigma=None
        # Loop for charge and discharge cycles with the parameters computed at the first cycle
        for cycle in df['Cycle'].unique():
            df_cycle = df[(df["Cycle"] == cycle) & (df["State"] == charging_state)]

            if len(df_cycle) > 3:
                df_smoothed, sigma = calculate_dqdv_for_one_cycle(df_cycle, 
                                                                  smoothing=smoothing, 
                                                                  sigma=sigma)
                df_smoothed = df_smoothed.assign(Cycle=cycle, State=charging_state)

                df_dqdv = pd.concat([df_dqdv, df_smoothed])
            else:
                print('No data for cycle '+str(cycle)+', '+str(charging_state))

    if len(df_dqdv) > 3:
        volt_step = len(df['Cycle'].unique()) * (df_dqdv['smoothed_voltage'].max() - df_dqdv['smoothed_voltage'].min()) / 50000

        df_dqdv = remove_low_dqdv_values(df_dqdv, 0.025)
        df_dqdv = create_linspace_voltage(df_dqdv, volt_step)  # 1e-3

        return df_dqdv
    
    else:
        print('Not possible to calculate dQ/dV')
        return False
    
    
def calculate_dqdv_for_one_cycle(df, smoothing=True, sigma=None):
    """Calculate dQ/dV for a specific cycle from a dataframe using columns 'Voltage' and 'Capacity'.

    Parameters
    -------
    df : pandas.DataFrame
        DataFrame containing the data of a specific cycle with the columns 'Voltage' and 'Capacity'
    smoothing : boolean, optional
        Whether to smooth the dQ/dV curve
    sigma: float, optional
        Standard deviation of the Gaussian filter to use for smoothing to control manually the smoothing

    Returns
    -------
    df_smoothed : pandas.DataFrame
        DataFrame containing the smoothed dQ/dV curve
    sigma : float
        Sigma parameter used for the smoothing
    """
    df_smoothed = df.sort_values(by='Voltage'
                                 ).drop_duplicates(subset='Voltage'
                                                   ).reset_index(drop=True
                                                                 ).copy()

    # Add data points by interpolating with steps of 0.1mV for short datasets
    if abs(df_smoothed['Voltage'].diff().min()) > 1e-3 and smoothing:
        df_smoothed = df_smoothed.drop_duplicates(subset='Voltage')
        f = interp1d(
                df_smoothed['Voltage'],
                df_smoothed['Capacity'],
                kind='cubic',
                )
        min_voltage, max_voltage = df_smoothed['Voltage'].min(), df_smoothed['Voltage'].max()
        voltage_linspace = np.linspace(min_voltage, max_voltage, int((max_voltage - min_voltage) / 1e-4))

        df_smoothed = pd.DataFrame({
            'Voltage': voltage_linspace,
            'Capacity': f(voltage_linspace)
        })
    
    # Rounding Voltage to 1mV and taking the median Capacity values
    df_smoothed['Voltage'] = round(df_smoothed['Voltage'], 3)
    df_smoothed = df_smoothed.groupby('Voltage', as_index=False
                                    ).agg({'Capacity': 'median'}
                                            ).reset_index(drop=True)

    df_smoothed['dqdv'] = np.gradient(df_smoothed['Capacity'], df_smoothed['Voltage'])

    df_smoothed[['dqdv']] = df_smoothed[['dqdv']].fillna(0)
    df_smoothed = df_smoothed.replace([np.inf, -np.inf], np.nan
                                      ).dropna(subset=['dqdv']
                                               ).reset_index(drop=True)

    df_smoothed = df_smoothed.assign(
        smoothed_voltage = df_smoothed['Voltage'],
        smoothed_capacity = df_smoothed['Capacity'],
        smoothed_dqdv = df_smoothed['dqdv']
        )
    
    # Remove outliers at the beginning and end of the cycle
    derivative_dqdv = np.diff(np.abs(df_smoothed['smoothed_dqdv']))
    idx_start, idx_end = np.where(derivative_dqdv >= 0)[0], np.where(derivative_dqdv <= 0)[0]
    if len(idx_start) > 0 and len(idx_end) > 0:
        idx_start = idx_start[0]
        idx_end = idx_end[-1]
        df_smoothed = df_smoothed.iloc[idx_start+1:idx_end]    

    if smoothing:
        if sigma:
            # Smooth the curve with the parameter sigma if given in input
            df_smoothed['smoothed_dqdv'] = gaussian_filter1d(df_smoothed['dqdv'], 
                                                             sigma=sigma)
            return df_smoothed, sigma

        else:
            # Find the height and width of the largest peak
            peaks, properties = find_peaks(abs(df_smoothed['dqdv']) / abs(df_smoothed['dqdv']).max(), 
                                           height=0.15)
            if len(peaks) > 0:
                largest_peak_index = np.argmax(properties['peak_heights'])
                largest_peak = peaks[largest_peak_index]

                height = properties['peak_heights'][largest_peak_index]
                width = peak_widths(abs(df_smoothed['smoothed_dqdv']), [largest_peak], rel_height=0.5)
                fwhm = width[0][0] * 1e-3
                
                # Smooth the curve with the parameter sigma based on the peak width and height
                r = fwhm / height
                sigma = max(1, np.log(r) + 7.5)
                df_smoothed['smoothed_dqdv'] = gaussian_filter1d(df_smoothed['dqdv'], 
                                                                 sigma=sigma)
            else:
                print('No peak found')

            return df_smoothed, sigma
        
    else:
        # Raw dQ/dV curve
        return df_smoothed, sigma


def create_linspace_voltage(df, voltage_step):
    """Create a linspace voltage vector for the dataFrame to have a consistent voltage vector for each cycle
    
    Parameters
    -------
    df : pandas.DataFrame
        DataFrame containing the data
    voltage_step : float
        Step size for the linspace voltage vector

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the linspace voltage data
    """
    # Boundaries of the linspace vector
    min_voltage, max_voltage = df['smoothed_voltage'].min(), df['smoothed_voltage'].max()
    linspace_voltage = np.linspace(min_voltage, max_voltage, int((max_voltage - min_voltage) / voltage_step))

    # Interpolating the dqdv and capacity values for each cycle and filling the missing values with the first and last values
    df_linspace = pd.DataFrame()
    for charging_state in df['State'].unique():
        for cycle in df['Cycle'].unique():

            df_cycle = df[(df['Cycle'] == cycle)
                          & (df['State'] == charging_state)
                          ].sort_values(by='smoothed_voltage')
            
            if len(df_cycle) > 3:
                df_linspace_cycle = pd.DataFrame({
                    'smoothed_voltage': linspace_voltage,
                    'Cycle': cycle,
                    'State': charging_state
                })

                for var in ['smoothed_dqdv', 'smoothed_capacity']:
                    f = interp1d(df_cycle['smoothed_voltage'], 
                                df_cycle[var], 
                                kind='linear', 
                                bounds_error=False, 
                                fill_value=(df_cycle[var].iloc[0], 
                                            df_cycle[var].iloc[-1]))
                    df_linspace_cycle[var] = f(linspace_voltage)

                df_linspace = pd.concat([df_linspace, df_linspace_cycle])

    return df_linspace


def remove_low_dqdv_values(df_input, thresh):
    """Remove the low dqdv values from the DataFrame 
    
    Parameters
    -------
    df_input : pandas.DataFrame
        DataFrame containing the data
    thresh : float
        Threshold for the low dqdv values

    Returns
    -------
    pandas.DataFrame
        DataFrame without the low dqdv values
    """
    df = df_input.copy()

    for charging_state in df['State'].unique():
        df_state = df[df['State'] == charging_state].copy()
        df_state['smoothed_dqdv'] = df_state['smoothed_dqdv'].abs()

        max_charge = df_state.groupby('Cycle')['smoothed_dqdv'].max().min()
        min_charge = df_state.groupby('Cycle')['smoothed_dqdv'].min().min()

        threshold_value = min_charge + (max_charge - min_charge) * thresh
        df = df[((df['smoothed_dqdv'].abs() >= threshold_value) | (df['State'] != charging_state))]

    return df