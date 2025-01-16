# BatteryDataAnalysis
A repository for battery data analysis tools.  

**BatteryDataAnalysis** is a Python package designed for analyzing battery data.  
It provides tools for processing, analyzing, and visualizing battery data, helping researchers and engineers to understand battery performance and behavior.

## Features
For the moment, the features available are:  

- **dQ/dV Analysis**: Calculate and analyze differential Capacity vs. Voltage (dQ/dV) curves.
- **Visualization**: Generate interactive plots for visualizing battery data, including dQ/dV plots, heatmaps and POCV curves.

## Usage

To correctly use this package, few steps are important:  

- Import the data in parquet format
- The data should contain at least the columns: *Current*, *Voltage*, *Time* and *Cycle*
- Other columns such as: *State* (charging state) and *Capacity* are still useful if available  
(Giving the column names in input is optional but helps the process)  

The **output** is a dataframe containing *Capacity*, *Voltage* and *dQ/dV*, and the choosen plots among dQ/dV curve, dQ/dV heatmap and POCV curve.  
An example notebook containing the different features of this package is given: *test/example.ipynb*

The usage of the main function:  

```python
from battmoanalysis.processing import process_dqdv

df = process_dqdv(file_path)
```

## Requirements

To install the required packages, you can use the following command:  

```bash
pip install -r requirements.txt
```