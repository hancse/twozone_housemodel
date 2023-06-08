housemodel
==========

|PyPI| |PyPI - Downloads| |Documentation Status| |PyPI - License|

A Python package for modelling the thermal behaviour of a building with HVAC installations.
The code is being developed in the Research Group Balanced Energy Systems (BES) at HAN University of Applied Sciences,
The ``housemodel`` package can be used under the conditions of the GPLv3 license.
|vspace| |br|

Installation
------------

This package can be installed using ``pip install housemodel``. This should
automatically install the dependency packages ``matplotlib`` , ``scipy``
and ``pandas``, if they haven't been installed already. If you are
installing by hand, ensure that these packages are installed as well.
|vspace| |br|

Example use
-----------

.. code:: python

   """Example Python script using the EduMUD module."""

   import numpy as np

   from edumud.constants import *
   from edumud.electrolytes import Electrolyte
   from edumud.particles import Particle
   from edumud.file_utils import load_config

   import matplotlib
   import matplotlib.pyplot as plt
   from pathlib import Path

   matplotlib.use("Qt5Agg")
   DATADIR = Path(__file__).parent

   el = Electrolyte(str(DATADIR / "cegm.yaml"))
   el.calc_kappa()

   param = load_config(str(DATADIR / "cegm.yaml"))
   part = Particle.from_dict(param["particles"][0])

   zeta_zero = BOLTZ_T / (el.z_plus * E_CHARGE)
   print(f"zeta_zero: {1000 * zeta_zero} mV")
   psi_zero = 0.5 * zeta_zero

|vspace| |br|

EduMUD pages
------------

-  `Pypi <https://pypi.org/project/edumud/>`__: EduMUD Python package
-  `BitBucket <https://bitbucket.org/deltares/edumud/>`__: EduMUD source
   code
-  `ReadTheDocs <https://edumud.readthedocs.io/>`__: EduMUD
   documentation
   
|vspace| |br|

Author and license
------------------

-  Author: Paul J.M. van Kan
-  Contact: http://vankanscientific.nl
-  License: `GPLv3 <https://www.gnu.org/licenses/gpl.html>`__

|vspace| |br|

References
----------

-  Data, API key and API documentation can be obtained from
   `Meteoserver.nl <https://meteoserver.nl/>`__

.. |PyPI| image:: https://img.shields.io/pypi/v/edumud?
.. |PyPI - Downloads| image:: https://img.shields.io/pypi/dm/edumud
.. |Documentation Status| image:: https://readthedocs.org/projects/edumud/badge/?version=latest
   :target: https://edumud.readthedocs.io/en/latest/?badge=latest
.. |PyPI - License| image:: https://img.shields.io/pypi/l/edumud?

.. |vspace| raw:: latex

   \vspace{5mm}

.. |br| raw:: html

   <br />