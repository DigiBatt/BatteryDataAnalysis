# BatteryDataAnalysis
A repository for battery data analysis tools.  

**BatteryDataAnalysis** is a Python package designed for analyzing battery data.  
It provides tools for processing, analyzing, and visualizing battery data, helping researchers and engineers to understand battery performance and behavior.

## Features
The features available are:  

- **Battery tests analysis**: Analyze the different test type and for each, extract parameters of the battery (such as diffusion coefficient, internal resistance...). The different test types are slow CCCV (for dQ/dV analysis), GITT, ICI, HPPC.
- **Visualization**: Generate interactive plots for visualizing battery data, including plots of parameters over the State of Charge and Voltage, Current and Capacity of the tests over time.

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
from src.processing import process_file

df, result_dict = process_file(file_path)
```

## Requirements

To install the required packages, you can use the following command:  

```bash
pip install -r requirements.txt
```