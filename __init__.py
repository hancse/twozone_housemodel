''' from python 3.3+ init.py does not neccessary anymore
init.py : used to be a required part of a package  
(old, pre-3.3 "regular package", not newer 3.3+ "namespace package").
'''
name = "house_model"  # why do we need packages name?

from .house 			  import *
from .qsun  			  import *
from .internal_heat_gain  import *
from .read_NEN 		      import *
from .Temperature_SP 	  import *
from .Total_Irrad 		  import *
from .parameters 		  import *
from .Simulation          import *


