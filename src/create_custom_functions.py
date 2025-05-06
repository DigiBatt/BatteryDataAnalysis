import inspect

def add_function(my_func, **kwargs):
    """Adds a function to the calculation process

    The user can create his own function to calculate a new parameter from the points given in the 
    :func:`pulse_number_GITT <BatteryDataAnalysis.src.analysis_GITT.pulse_number_GITT>`, 
    :func:`pulse_number_ICI <BatteryDataAnalysis.src.analysis_ICI.pulse_number_ICI>` 
    or :func:`pulse_number_HPPC <BatteryDataAnalysis.src.analysis_HPPC.pulse_number_HPPC>` documentation.

    The new custom function must be an argument of :func:`process_file <BatteryDataAnalysis.src.processing.process_file>`, as detailled in the example.

    Then, the new parameter is calculated and stored in the result DataFrame and plotted over SOC.
    
    Parameters
    ----------
    my_func : function
        Function to add to the calculation process
    **kwargs : dict
        Keyword arguments to pass to the function.  
        The parameters available are:  

        - Specific parameters defined in :func:`pulse_number_GITT <BatteryDataAnalysis.src.analysis_GITT.pulse_number_GITT>`, 
        :func:`pulse_number_ICI <BatteryDataAnalysis.src.analysis_ICI.pulse_number_ICI>` 
        or :func:`pulse_number_HPPC <BatteryDataAnalysis.src.analysis_HPPC.pulse_number_HPPC>` : float values,

        - Voltage, Current and Capacity : interpolation functions of time,

        - df_pulse : DataFrame containing the pulse's data,

        - State : 'C' or 'D' corresponding to the charging state of the pulse.

    Returns
    -------
    Float
        New parameter calculated by the function my_func

    Examples
    --------
    >>> def my_example_func(Ipulse, Voltage, t0):
    ...     Internal_resistance = (Voltage(t0 + 1) - Voltage(t0)) / Ipulse
    ...     return Internal_resistance
    >>> df = process_file(file_path, my_func_list=[my_example_func])
    File : GITT_AG4_S_1577.parquet
    Length : 9154638
    Preprocessing Time : 41s
    Find Test Time : 48s
    Test : GITT ; Length : 9061575
        Pulse  Cycle   TestTime       SOC       OCV    Ipulse  Diffusion Coefficient  Resistance      R_1s      R_30s      R_60s     R_180s      Tau    State  my_example_func
    Pulse      0      1   46801.00  1.000000  2.484417 -0.043325           3.224942e-15   24.730671  6.169087  28.370994  30.130583  36.914223  3598.98     D         6.169087
    Pulse      1      1   53999.98  0.980011  0.575001 -0.043325           4.672011e-16    1.997885  1.474679   2.089985   2.411197   3.278645  3599.98     D         1.474679
    Pulse      2      1   61199.98  0.960011  0.379431 -0.043325           3.873728e-16    1.412661  1.106010   1.468100   1.685784   2.282023  3599.98     D         1.106010
    Pulse      3      1   68399.98  0.940010  0.272730 -0.043324           2.772498e-16    1.205467  0.979083   1.249654   1.411445   1.843224  3599.98     D         0.979083
    Pulse      4      1   75599.98  0.920010  0.216829 -0.043325           6.646707e-17    1.077636  0.915281   1.112744   1.224103   1.524546  3599.98     D         0.915281
    ...      ...    ...        ...       ...       ...       ...                    ...         ...       ...        ...        ...        ...      ...   ...              ...
    Pulse    133      2  962099.98  0.136391  0.087613 -0.043325           1.887492e-19    0.220889  0.143877   0.239615   0.273980   0.321392  3599.98     D         0.143877
    Pulse    134      2  969299.98  0.111716  0.087493 -0.043325           2.520212e-19    0.225816  0.147635   0.244767   0.279301   0.326996  3599.98     D         0.147635
    Pulse    135      2  976499.98  0.087041  0.087352 -0.043325           3.536782e-19    0.231572  0.152126   0.251083   0.285541   0.333647  3599.98     D         0.152126
    Pulse    136      2  983699.98  0.062366  0.087180 -0.043325           5.092976e-19    0.238380  0.157598   0.258096   0.292359   0.341250  3599.98     D         0.157598
    Pulse    137      2  990899.98  0.037691  0.086971 -0.043325           8.648767e-19    0.246495  0.163909   0.266569   0.300817   0.350488  3599.98     D         0.163909
    [138 rows x 15 columns]
    GITT Time : 21s
    Total Time : 115s
    """
    sig = inspect.signature(my_func)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    for k in sig.parameters:
        if k not in filtered_kwargs:
            return None
    return my_func(**filtered_kwargs)