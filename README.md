# AeroelasticAirfoil-SU2
A simple framework that automatically generates SU2 configuration (.cfg) files and make calls to SU2 to perform aeroelastic simulations all within Python. The Matrix Pencil [4] method is then used to compute the damping coefficient. The code is based on SU2 v7.2.1 Blackbird (any version >= 7 should work).  I wrote this code for my undergraduate thesis "On a Support Vector Machine Approach To Surrogate Modelling: Predicting the Aeroelastic Flutter Boundary Within the Transonic Regime" alongside a conference paper [2].


Notes:

    Required packages: numpy, scipy

References:

[1] Economon, T. D., Palacios, F., Copeland, S. R., Lukaczyk, T. W., and Alonso, J.J., SU2: An Open-Source Suite for Multiphysics Simulation and Design, AIAA Journal, Vol. 54, No. 3, 2016, pp. 828–846. (https://github.com/su2code/SU2)

[2] Palar, P.S., Izzaturahman, F., Zuhal, L.R. and Shimoyama, K., Prediction of the Flutter Boundary in Aeroelasticity via a Support Vector Machine.

[3] Palar, P.S., Parussini, L., Bregant, L., Shimoyama, K., Izzaturrahman, M.F., Baehaqi, F.A. and Zuhal, L., 2022. Composite Kernel Functions for Surrogate Modeling using Recursive Multi-Fidelity Kriging. In AIAA SCITECH 2022 Forum (p. 0506).

[4] Jacobson, K.E., Kiviaho, J.F., Kennedy, G.J. and Smith, M.J., 2019. Evaluation of time-domain damping identification methods for flutter-constrained optimization. Journal of Fluids and Structures, 87, pp.174-188.
