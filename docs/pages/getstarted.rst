Quick Start
===========

Welcome to the Getting Started Guide for **BatteryDataAnalysis**.

BatteryDataAnalysis is a Python tool designed to help you process and analyze battery test data.
It supports data in CSV or Parquet formats containing voltage, current, and time series.

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

   For more details, see the :ref:`installation` section.

2. **Import the library** in your Python scripts or Jupyter notebooks:

   .. code-block:: python

       import BatteryDataAnalysis

3. **Start analyzing your data**:

   .. code-block:: python

       from BatteryDataAnalysis.src.processing import process_file

       df, result_dict = process_file("path/to/your/data.csv")


Exploring Examples
------------------

To help you get acquainted with BatteryDataAnalysis's capabilities, we provide a collection of examples that demonstrate common use cases and features of the package:

- **Jupyter Notebooks**: Interactive notebooks that include detailed explanations alongside the live code, visualisations, and results. These are an excellent resource for learning and can be easily modified and executed to suit your needs.

- **Python Scripts**: For those who prefer working in a text editor, IDE, or for integrating into larger projects, we provide equivalent examples in plain Python script format.

You can find these resources in the ``examples`` folder of the BatteryDataAnalysis repository. To access the examples, navigate to the following path after cloning or downloading the repository:

.. code-block:: console

    path/to/BatteryDataAnalysis/docs/examples

# These examples are also available on our `GitHub repository <https://github.com/DigiBatt/BatteryDataAnalysis>`_.


Support and Contributions
-------------------------

If you encounter any issues or have questions as you start using PyBOP, don't hesitate to reach out to our community:

- **GitHub Issues**: Report bugs or request new features by opening an `Issue <https://github.com/DigiBatt/BatteryDataAnalysis/issues>`_
- **GitHub Discussions**: Post your questions or feedback on our `GitHub Discussions <https://github.com/DigiBatt/BatteryDataAnalysis/discussions>`_
- **Contributions**: Interested in contributing to BatteryDataAnalysis? Check out our `Contributing Guide <Contributing.html>`_ for guidelines.