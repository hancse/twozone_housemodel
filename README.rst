==========
housemodel
==========

|PyPI| |Documentation Status| |PyPI - License|

A Python package for modelling the thermal behaviour of a building with HVAC installations.
The code is being developed in the Research Group Balanced Energy Systems (BES) at HAN University of Applied Sciences,
The ``housemodel`` package can be used under the conditions of the GPLv3 license.

Features
--------

* Basic classes for thermal network representation of buildings.
* Additional classes for HVAC installations.
* Conductive and convective heat transfer.
* Matrix formalism for dynamic evaluation of temperature distributions.
* Heat rate balance equations including solar irradiance and human presence.
* ODE solver evaluating building dynamics and HVAC control.


Installation
------------

To use the package `housemodel`, install it in a Python environment:

.. code-block:: console

    (env) pip install housemodel

or

.. code-block:: console

    (env) conda install housemodel

This should
automatically install the dependency packages ``matplotlib`` , ``scipy``
and ``pandas``, if they haven't been installed already. If you are
installing by hand, ensure that these packages are installed as well.

Example use
-----------

.. code:: python

   """Example Python script using the housemodel package."""

   from housemodel.buildings.building import Building

    # create Building object
    h = Building("MyHouse")
    section = param["Building"]
    # read nodes attribute from dictionary and create capacity matrix
    h.nodes_from_dict(section["nodes"])
    h.fill_c_inv()
    h.edges_from_dict(section["edges"])
    # read FixedNode objects (external nodes);
    h.boundaries_from_dict(param["boundaries"])  # function selects "outdoor" as ambient
    h.make_k_ext_and_add_ambient()  # initialize k_ext_mat and add diagonal elements

    b = StratifiedBufferNew.from_dict(param["Buffer"])
    b.generate_nodes()
    b.fill_c_inv()
    b.generate_edges()
    b.generate_ambient()
    b.make_k_ext_and_add_ambient()

    total = TotalSystem("HouseWithBuffervessel", [h, b])
    total.sort_parts()
    # compose c-1-matrix from parts and merge tag_lists
    total.merge_c_inv()
    total.merge_tag_lists()
    ...


housemodel pages
----------------

-  `PyPi <https://pypi.org/project/housemodel/>`__: housemodel Python package
-  `Github <https://github.com/hancse/twozone_housemodel>`__: housemodel source
   code
-  `ReadTheDocs <https://housemodel.readthedocs.io/>`__: housemodel
   documentation

Author and license
------------------

-  Author: Paul J.M. van Kan
-  Contact: paul.vankan@han.nl
-  License: `GPLv3 <https://www.gnu.org/licenses/gpl.html>`__

References
----------

- ...

.. |PyPi| image:: https://img.shields.io/pypi/v/housemodel
   :alt: PyPI

.. |PyPI - Downloads| image:: https://img.shields.io/pypi/dm/housemodel
   :alt: PyPI - Downloads

.. |PyPi Status| image:: https://img.shields.io/pypi/status/housemodel
   :alt: PyPI - Status

.. |Documentation Status| image:: https://readthedocs.org/projects/housemodel/badge/?version=latest
   :target: https://edumud.readthedocs.io/en/latest/?badge=latest

.. |PyPI - License| image:: https://img.shields.io/pypi/l/housemodel
   :alt: PyPI - License
