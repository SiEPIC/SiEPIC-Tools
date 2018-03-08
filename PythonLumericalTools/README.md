
# Circuit simulations from Netlist file:
* We have a Python script that reads the netlist file and performs a simulation in Lumerical INTERCONNECT.
* It can be executed in any Python environment, including:
  * Python Jupyter web-based notebook
  * Python in KLayout
* Instructions below for these options

## using Python Jupyter

### Installation instructions for Python Jupyter:
* Jupyter is a web-browser environment for Python scripting
* Installation instructions: http://jupyter.readthedocs.io/en/latest/install.html
  * Windows: easiest is to install Anaconda 3.6 [2 GB]
  * MacOS (similar in Linux): Anaconda, or an easy alternative [~ 100 MB] is to install using brew and pip in Terminal.App:
* Install the package manager: https://brew.sh
     /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
* Install Python 3
     brew install python3
* Install Jupyter notebook
     pip3 install jupyter
     pip3 install numpy
  * Launch the web-browser notebook:
     jupyter notebook
* In the Jupyter web browser window, find the Python file to execute.

### Circuit simulations from Netlist file, using Python Jupyter and Lumerical FDTD:
* In the Jupyter web browser window, find the Python notebook file to execute: PythonLumericalTools/montecarlo_lumerical.ipynb
* Choose which netlist file to load, e.g., “MZI_bdc” or “RingResonator”, by commenting out the other ones.
* Select the first cell, then click “Run”. Repeat one by one for all of them.
* You should get a simulation result.  
* For Monte Carlo simulations, change the “if 0” to “if 1” after “# Monte Carlo simulation”

## using Python in KLayout

### Installation
* http://www.klayout.de/build.html -- Download current version

### Circuit simulations from Netlist file, using Python in KLayout and Lumerical FDTD:
* F5 for Macro Development window
* Click on the “Python” tab at the top left
* In the left panel, right-click and Add Location
* Find the folder PythonLumericalTools, then Open
* Expand the added Project folder and double-click on “montecarlo_lumerical”
* Click on the run! button (green play with exclamation button)
* You should get a simulation result.
