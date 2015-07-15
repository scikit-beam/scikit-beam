Differential Phase Contrast (DPC) imaging demo
==============================================

What it includes
----------------

dpc_demo.py: an example script for conducting DPC using functional modules 
             in dpc.py
             
dpc_demo.ipynb : An example notebook for analyzing DPC data [link] 
                 (https://github
                 .com/Nikea/scikit-xray-examples/blob/master/demos/dpc/dpc_demo.ipynb)

a.jpg, phi.jpg: final results of dpc demo script after processing files in 
                SOFC/ directory

                  

dpc_demo.py
-----------
This script downloads an example data set (if it is not already present), 
sets up some state and calls ``skxray.dpc.dpc_runner`` to conduct an example 
DPC calculation using the data set that was published in (not sure where to 
find that paper...)

It requires an input file folder containing diffraction patterns that is 
currently located in the ``SOFC/`` directory next to ``dpc_demo.py``. If the 
``SOFC/`` directory does not exist, the data will be automatically downloaded 
and extracted to ``SOFC/``. After these files have been downloaded and 
extracted, the example code will run.  This download will only occur if the 
``SOFC/`` directory is not present.

The input files can be manually downloaded [here] 
(https://www.dropbox.com/s/963c4ymfmbjg5dm/SOFC.zip"). Please extract it 
to the directory where the ``dpc_demo.py`` file is located.


Output images
-------------
**phase**: The final reconstructed phase image.
![phase.jpg](https://www.github.com/scikit-xray-examples/demos/dpc/phase.jpg)

**amplitude.jpg**: Amplitude of the sample transmission function.
![amplitude.jpg](https://www.github.com/scikit-xray-examples/demos/dpc/amplitude.jpg)

