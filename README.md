# MultiParameterContinuation
This project is meant to be used as a prototype for a multiparameter continuation method.

## Description

This project explore two multiparameter continuation methods applied to systems of equations with manifolds constraints. The methods are applied on two positioning problems with an implementation in Python, using the library Pymanopt.

## Getting Started

### Dependencies

You will need to download the library `pymanopt` which can be found [here](https://www.pymanopt.org/) as well as the library `autograd` [here](https://github.com/HIPS/autograd).

### Code description

* The folder `Continuation` contains all the files required to perform the multiparameter continuation. The prediction on the solution manifold's tangent space is performed either using the hessian of a related cost function, or the differential of the system (please refer to PdM report for more details). The class `PathAdaptiveMultiParameterContinuation` which performs the ellipsoid method can be found in the file `Continuation\PositioningProblem\PerturbationWithDifferential\PathAdaptiveContinuation.py`.
* The folder `Helpers` contains basic SE(3) transformations as well as methods to express linear operators in basis of SE(3) x R^N.
* The folder `PositioningProblems` defines the different studied problems, as well as a related cost function and the system's differential for each problem.
* The folder `NumericalExperiments` contains different numerical experiments that has been done:
  * `NumericalExperiments\FeasibleMap` allows to compute and plot the feasible map of parameters for the parabola test case with two dimensional parameter space.
  * `NumericalExperiments\MetricAtBorder` allows to compute and plot the metric of the ellipsoid method at the border of the feasible region of parameters for parabola test case with two dimensional parameter space.
  * `NumericalExperiments\StatisticsOnProblems` allows to run statistics in the same way as in the PdM report, for the parabola test case with three dimensional parameter space and the torus-helices test case.
  * `NumericalExperiments\TestCases` allows to run the parabola and torus-helices test cases for a default initial parameter (if you want to run with other parameters, just find a numerical solution using the default parameters, then change the variables `initialParameter` and `initialSolution` accordingly with your results). Visualization is provided after the solution is computed.
  * `NumericalExperiments\ToleranceSearchForAdaptiveMethod` allows to find the best tolerance for a given test case.
* The folder `SE3Parameterizations` contains all the SE(3) parameterizations which have been used for the project (helix, parabola and torus)
* `Solver` contains the RBFGS method for positioning problems and eigenvalue problem. Also it contains a first version of the GLTR algorithm.
* Finally, the folder `Visualization` contains all the tools needed for the visualization of results in `NumericalExperiments\TestCases`.
