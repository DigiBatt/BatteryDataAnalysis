Quick Start
===========

Welcome to the Getting Started Guide for **BatteryDataAnalysis**.

BatteryDataAnalysis is a Python tool designed to help you process and analyze battery test data.
It supports data in csv, parquet, txt and excel formats containing voltage, current, and time series.

The package includes analysis tools for tests such as:

- GITT (Galvanostatic Intermittent Titration Technique)
- ICI (Intermittent Current Interruption)
- HPPC (Hybrid Pulse Power Characterization)
- dQ/dV (Differential Capacity Analysis)

Getting Started with BatteryDataAnalysis
----------------------------------------

To begin using **BatteryDataAnalysis**, follow these steps:

1. **Clone the repository locally** (if you want the source version):

   .. code-block:: console

       git clone https://github.com/DigiBatt/BatteryDataAnalysis.git
       cd BatteryDataAnalysis
       pip install .
       pip install -r requirements.txt

   This will install the package and its dependencies on your local environment.

2. **Import the library** in your Python scripts or Jupyter notebooks:

   .. code-block:: python

       import BatteryDataAnalysis

Start analyzing your data
-----------------------------

To analyze a single file:

.. code-block:: python

    from BatteryDataAnalysis import process_file

    df, result_dict = process_file("path/to/your/data.csv")

Supported file formats: `.csv`, `.parquet`, `.txt`, `.xlsx` and `.xls`

The file must contain at least the following columns:

- SysTime (which can be system time or test time)
- Voltage
- Current

Optionally, you can include a `column_names.json` file **in the same folder** as your data file to specify custom column names:

.. code-block:: json

    {
        "time_column_name": "SysTime",
        "voltage_column_name": "Voltage",
        "current_column_name": "Current"
    }

You can also include a `debug_func.py` file **in the same folder** as your data to preprocess or modify the data before analysis. It must define a function `debug_func(df)` that returns a modified version of the DataFrame:

.. code-block:: python

    # debug_func.py
    def debug_func(df):
        # Example: keep only the first 1e6 seconds of data
        df = df[df['time_column_name'] < 1e6]
        return df

These optional files allow you to tailor the analysis to your specific data format or preprocessing needs.

You can also use a DataFrame as an input of the process_file function:

.. code-block:: python

    import pandas as pd
    from BatteryDataAnalysis import process_file

    # Load your data into a DataFrame
    df = pd.read_csv("path/to/your/data.csv")

    # Process the DataFrame directly
    df, result_dict = process_file(df, input='dataframe')


Customising the Analysis
-------------------------

BatteryDataAnalysis allows you to customise the data processing pipeline by providing your own functions to manipulate or clean the data before it is analyzed.

To do this, create a function in the same file as the process_file use. You should put it in a list called `my_func_list` and pass it to the `process_file` function.
The input parameters you can use are described in the create_custom_functions file in the :ref:`api-reference`.

Here is an example of a simple custom function for a GITT test:

.. code-block:: python

    def my_example_func(Ipulse, Voltage, t0):
    Internal_resistance = (Voltage(t0 + 1) - Voltage(t0)) / Ipulse
    return Internal_resistance

    df, result_dict = process_file("path/to/your/data.csv", my_func_list=[my_example_func])

This makes the framework flexible and adaptable to a wide variety of experimental setups and data quirks.


Exploring Examples
------------------

To help you get acquainted with BatteryDataAnalysis's capabilities, we provide a collection of examples that demonstrate common use cases and features of the package.

You can find these resources in the :ref:`examples-page` page, located in:

.. code-block:: console

    path/to/BatteryDataAnalysis/docs/examples

These examples are also available on our `GitHub repository <https://github.com/DigiBatt/BatteryDataAnalysis>`_.


Support and Contributions
-------------------------

If you encounter any issues or have questions as you start using BatteryDataAnalysis, don't hesitate to reach out to our community:

- **GitHub Issues**: Report bugs or request new features by opening an `Issue <https://github.com/DigiBatt/BatteryDataAnalysis/issues>`_
- **GitHub Discussions**: Post your questions or feedback directly to the project founders that you can find on `GitHub <https://github.com/DigiBatt/BatteryDataAnalysis>`_
- **Contributions**: Interested in contributing to BatteryDataAnalysis? Check out our `Contributing Guide <Contributing.html>`_ for guidelines.