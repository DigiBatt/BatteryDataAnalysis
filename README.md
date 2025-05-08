# BatteryDataAnalysis

A repository for battery data analysis tools.  

**BatteryDataAnalysis** is a Python package designed for analyzing battery data.  
It provides tools for processing, analyzing, and visualizing battery data, helping researchers and engineers to understand battery performance and behavior.

## Features

The features available are:  

- **Battery tests analysis**: Analyze the different test type and for each, extract parameters of the battery (such as diffusion coefficient, internal resistance...). The different test types are slow CCCV (for dQ/dV analysis), GITT, ICI, HPPC.
- **Visualization**: Generate interactive plots for visualizing battery data, including plots of parameters over the State of Charge and Voltage, Current and Capacity of the tests over time.

## Installation

To install the package locally, you have to clone the repository before installing the package:

```bash
git clone https://github.com/DigiBatt/BatteryDataAnalysis.git
cd BatteryDataAnalysis
pip install .
```

This will install the package and its dependencies on your local environment.

## Usage

To correctly use this package, few steps are important:  

- Import the data in parquet, excel, csv or txt formats.
- The data should contain at least the columns: *Current*, *Voltage* and *Time*. If their column names are not easily recognizable, it is possible to add it in input of the processing function to help the process.  

The **outputs** are:

- A dataframe *df* containing the preprocessed data,
- A dataframe *result_dict* containing the extracted parameters for each SOC and cycle
- The result plots displaying the results and the test over time.  
An example notebook containing the different features of this package is given: *test/example.ipynb*

The usage of the main function:  

```python
from BatteryDataAnalysis import process_file

df, result_dict = process_file(file_path)
```

## Documentation

To build the sphinx documentation, you can follow these steps:

```bash
pip install -r requirements-docs.txt
cd .\docs\
.\make.bat clean
.\make.bat html
```

You can then display the documentation on your browser by running the HTML file at the path:  
C:/BatteryDataAnalysis/docs/_build/html/index.html
