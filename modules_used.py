# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 08:21:28 2022

@author: Mani
These are the usual modules that I need for analysis. To bring all the libaries into the 
console, type in:
    from modules_used import *
    
"""

mod_list = ['ripples:ripples',
            'cont:cont',
            'rip_data_processing:rdp',
            'rip_data_plotting:rdpl',
            'djutils:dju',
            'general_functions:gf',
            'pandas:pd']
import importlib
import re
print('Imported:')
for libstr in mod_list:
    ss = re.findall('(.+):(.+)', libstr)[0]
    globals()[ss[1]] = importlib.import_module(ss[0])
    print(f'{ss[0]} as---------- {ss[1]}')

