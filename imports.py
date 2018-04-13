shortcuts = {}#{'/Users/zkbt/Cosmos/Data/Magellan/LDSS3':'...'}
import craftroom.Talker
craftroom.Talker.shortcuts = shortcuts
#craftroom.Talker.line = 200
Talker = craftroom.Talker.Talker

import craftroom.Writer
Writer = craftroom.Writer.Writer

import astropy.io.fits, astropy.io.ascii, astropy.time
from astropy import time, coordinates as coord, units as u

import matplotlib.pyplot as plt, numpy as np, scipy

import os
