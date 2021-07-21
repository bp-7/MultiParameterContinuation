# MultiParameterContinuation
This project is meant to be used as a prototype for a multiparameter continuation method.

## Description

This project explore two multiparameter continuation methods applied to systems of equations with manifolds constraints. The methods are applied on two positioning problems with an implementation in Python, using the library Pymanopt.

## Getting Started

### Dependencies

You will need to download the library `pymanopt` which can be found [here](https://www.pymanopt.org/) as well as the library `autograd` [here](https://github.com/HIPS/autograd).

### Executing program

* The folder `Continuation` contains all the files required to perform the multiparameter continuation. The prediction on the solution manifold's tangent space is performed either using the hessian of a related cost function, or the differential of the system (please refer to PdM report for more details). The class `PathAdaptiveMultiParameterContinuation` which performs the ellipsoid method can be found in the file `Continuation\PositioningProblem\PerturbationWithDifferential\PathAdaptiveContinuation.py`.
* The folder `Helpers` contains basic SE(3) transformations as well as methods to express linear operators in basis of SE(3) x R^N.
* The folder `PositioiningProblems` defines the different studied problems, as well as a related cost function and the system's differential for each problem.
* The folder `NumericalExperiments` contains different numerical experiments that has been done:
  * In particular in `NumericalExperiments\FeasibleMap`,  
```
code blocks for commands
```
