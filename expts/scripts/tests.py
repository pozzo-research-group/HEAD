import numpy as np
import head
import matplotlib.pyplot as plt

from pyGDM2 import (core, propagators, fields, 
                    materials, linear, structures, 
                    tools, visu)

import os, shutil, time

for i in range(10):
	print(i, time.time())
	time.sleep(10)