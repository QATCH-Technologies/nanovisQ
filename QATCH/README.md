# QATCH

## Name
QATCH Q-1 GUI Software

## Programming language
Python

## Author
QATCH Team

## Description
An open-source Python application, to display, process and store data in real-time from the QATCH Q-1 Device.
The main functionality of the software is the real-time monitoring of frequency and dissipation variations
of a quartz crystal microbalance through the analysis of the resonance curve.
The application uses the multiprocessing package (https://docs.python.org/3/library/multiprocessing.html).
- includes internal and external packages

## Intended Audience
Science/Research/Engineering

## Software Development
User Interfaces

## Requirements
Requirements:
- Python 3.7 (verified compatibility with Python 3.6) (https://www.python.org/).
- Anaconda3-5.3.0
     External Packages:
     - PyQt5 (https://pypi.org/project/PyQt5/).
     - PySerial 3.4 (https://pypi.org/project/pyserial/).
     - PyQtGraph 0.10.0 (http://www.pyqtgraph.org/).
     - progressbar 2.5 (https://pypi.org/project/progressbar/).


## Other used internal packages:
- multiprocessing, numpy, scipy, setuptools, io, platform, sys, enum, argparse, cvs, time, datetime, logging, etc.

## Installation instructions/guide:
Using Anaconda3
Windows, macOS
  1.  Download and install Anaconda3 for Python 3.7 version Anaconda3-5.3.0  https://www.anaconda.com/download/
  2.  During Anaconda3 installation select the check mark shown in the figure below:

  3.  Open Anaconda3 prompt (Windows) or terminal (macOS) and type (install/upgrade Python packages) :
        conda install pyqtgraph pyserial
        python -m pip install --upgrade pip
        python -m pip install --upgrade h5py
        pip install progressbar

Linux
  1.  Type the command below by replacing username with that of your pc change permission of    
                Anaconda3  
        sudo chown -R username:username /home/username/anaconda3
  2.  Open Anaconda3 terminal  and type (install/upgrade Python packages) :
        conda install pyqtgraph pyserial
        pip install --upgrade pip --user
        pip install progressbar --user
  3.  Set permission on serial port
        sudo usermod -a -G uucp username
        sudo usermod -a -G dialout username
  4.  Logout and Login


## Usage
Start the application from Anaconda3 prompt
1.  Launch Anaconda3 prompt
2.  Browse to the QATCH Q-1 Python software main directory
    ...\QATCH_Q-1_py_v2.0\QATCH\
3.  Start the application main GUI by typing the command
pyhton – m QATCH

Start the application double-click app.py file
You can make executable and launch app.py Python file
1.    Browse to the QATCH Q-1 Python software main directory
      ...\QATCH_Q-1_py_v2.0\QATCH\
2.    Right click on app.py file -> open with -> choose another app in this PC
3.    Browse to Anaconda3 directory on your PC
      C:\Users\[your_user_name]\Anaconda3
4.    Select python.exe executable file
5.    Double-click on app.py file in the QATCH Python software main directory (see folders and program files below)

## Contact
- [mail] info@qatchtech.com

## License and Citations
The project is distributed under GNU GPLv3 (General Public License).
