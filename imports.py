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

import os, sys
sys.path.append('/home/hdiamond/local/lib/python2.7/site-packages/')
sys.path.append('/h/mulan0/code/')
sys.path.append('/h/mulan0/code/mosasaurus')
sys.path.append('/h/mulan0/code/decorrasaurus')
