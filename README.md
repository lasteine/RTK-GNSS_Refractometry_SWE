# SWE_RTK-GNSS
Snow water equivalent estimation based on RTK-GNSS (real-time-kinematic Global navigation satellite system).

The biased up-component of a short GNSS baseline between a base antenna (mounted on a pole) and a rover antenna (mounted underneath the snowpack) is used in this approach. Emlid Reach M2 receivers are used in a field setup, connected to ublox ANN-MB1 multi-frequency and multi-GNSS antennas. The receivers are set-up in RTK mode (base and rover settings) and the baseline vector (in ENU: East, North, Up) is logged by the receiver and stored in a .ENU file.

The python script contains a workflow from .ENU files reading to plotted SWE timeseries.


The method follows Steiner et al. (2020): 

L. Steiner, M. Meindl, C. Marty and A. Geiger, "Impact of GPS Processing on the Estimation of Snow Water Equivalent Using Refracted GPS Signals," in IEEE Transactions on Geoscience and Remote Sensing, vol. 58, no. 1, pp. 123-135, Jan. 2020, doi: 10.1109/TGRS.2019.2934016.
