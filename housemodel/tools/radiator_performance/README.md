# Radiator Return Temperature Calculator
This Python tool calculates the return temperature for a residential radiator unit as based on the performance/rating measures making use of various mean temperature difference approaches.  

## Table of Contents
- [Features](README.md#features)
- [How2Use](README.md#how2use)
- [License](README.md#license)
- [Acknowledgement](README.md#acknowledgement)
- [How2Cite](README.md#how2cite)
- [References](README.md#references)

## Features
The main idea here is to develop simple and reliable emprical models for residential radiator units so to be used in system level modelling (i.e. at district heating simulations). Hence, this Python models estimate the radiator behaviour in terms of return temperature at different operational conditions (e.g. at various rates of heat demand at different degrees of supply temperature).

- Three different Python functions are given, each basing on different approaches for the radiator excess temperature such as either Logarithmic Mean Temperature Difference (LMTD) - implicit, Geometric Mean Temperature Difference (GTMD) - explicit, or Arithmetic Mean Temperature Difference (AMTD) - explicit.
- Error evaluation are involved for GMTD and AMTD approaches, as error criteria formulated with the approach factor. 

## How2Use
An example script is given: [ExampleScript_RadiatorPerformance.py](https://github.com/DrTol/radiator_performance-Python/blob/master/ExampleScript_RadiatorPerformance.py), which illustrates how to use the Python functions developed, as based on AMTD, GMTD, and LMTD, all of which can be found in [ReturnTemperature.py](https://github.com/DrTol/radiator_performance-Python/blob/master/ReturnTemperature.py). 

Please clone or download the whole repository and run this example script! 

## License
You are free to use, modify and distribute the code as long as the authorship is properly acknowledged.  

## Acknowledgement 
We would like to acknowledge all of the open-source minds in general for their willing of share (as apps or comments/answers in forums), which has encouraged our department to publish these Python tools developed during the PhD study here in GitHub.

## How2Cite:
1. Tol, Hİ. radiator_performance-Python. DOI: 10.5281/zenodo.3265381. GitHub Repository 2019; https://github.com/DrTol/radiator_performance-Python
2. Tol, Hİ. District heating in areas with low energy houses - Detailed analysis of district heating systems based on low temperature operation and use of renewable energy. PhD Supervisors: Svendsen S. & Nielsen SB. Technical University of Denmark 2015; 204 p. ISBN: 9788778773685.

## References
- Phetteplace GE. Optimal design of piping systems for district heating. Hanover, N.H.: U.S. Army Cold Regions Research and Engineering Laboratory; 1995.
- Bøhm B. Energy-economy of Danish district heating systems: A technical and economic analysis. Lyngby, Denmark: Laboratory of Heating and Air Conditioning, Technical University of Denmark; 1988.
- Benonysson A. Dynamic modelling and operational optimization of district heating systems. Lyngby, Denmark: Laboratory of Heating and Air Conditioning, Technical University of Denmark; 1991.
- Heat Emission from Radiators and Heating Panels, link: https://www.engineeringtoolbox.com/heat-emission-radiators-d_272.html
- British Standards Institution. BS EN 442-2:2014: Radiators and convectors - Part 2: Test methods and rating 2014:82.
- Soumerai H. Practical thermodynamic tools for heat exchanger design engineers. New York: Wiley-Interscience; 1987.
- Radson. Kv-calculator_-_03-2012(1).xls 2012:2.
- Schlapmann D. Heat output and surface temperature of room heat emitters [Warmeleitung und oberflachentemperatureen von raumheizkorpern] (German). Heiz Luft Haustechnik 1976;27:317–21.
- Kilkis İB. Equipment oversizing issues with hydronic heating systems. ASHRAE J 1998;40:25–30.
