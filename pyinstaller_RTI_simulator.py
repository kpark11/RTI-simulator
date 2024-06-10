# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 23:42:07 2024

@author: brian
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 18:46:31 2024

@author: brian
"""

import PyInstaller.__main__

PyInstaller.__main__.run([
    'RTI_simulator.py',
    '--onefile',
    #'--onedir',
    '--noconfirm',
    '--clean',
    '--optimize',
    '1',
    '--hide-console',
    'hide-late', # Windows only
    #'--windowed',
    #'--console',
    #'--splash=icon.jpg',
    '--icon=icon.jpg',
    '--add-data=data_files\Sub10-1__result.mat;data_files',
    '--add-data=data_files\\Sub10-2__result.mat;data_files',
    '--add-data=data_files\\Sub1-1__result.mat;data_files',
    '--add-data=data_files\\Sub11-1__result.mat;data_files',
    '--add-data=data_files\\Sub11-2__result.mat;data_files',
    '--add-data=data_files\\Sub12-1__result.mat;data_files',
    '--add-data=data_files\\Sub12-2__result.mat;data_files',
    '--add-data=data_files\\Sub13-1__result.mat;data_files',
    '--add-data=data_files\\Sub13-2__result.mat;data_files',
    '--add-data=data_files\\Sub14-1__result.mat;data_files',
    '--add-data=data_files\\Sub14-2__result.mat;data_files',
    '--add-data=data_files\\Sub3-1__result.mat;data_files',
    '--add-data=data_files\\Sub3-2__result.mat;data_files',
    '--add-data=data_files\\Sub4-1__result.mat;data_files',
    '--add-data=data_files\\Sub4-2__result.mat;data_files',
    '--add-data=data_files\\Sub5-1__result.mat;data_files',
    '--add-data=data_files\\Sub5-2__result.mat;data_files',
    '--add-data=data_files\\Sub6-1__result.mat;data_files',
    '--add-data=data_files\\Sub7-1__result.mat;data_files',
    '--add-data=data_files\\Sub8-1__result.mat;data_files',
    '--add-data=data_files\\Sub8-2__result.mat;data_files',
    '--add-data=data_files\\Sub9-1__result.mat;data_files',
    '--add-data=data_files\\Sub9-2__result.mat;data_files',
    '--add-data=data_files\\RandomForestClassifier.pkl;data_files',
    '--add-data=data_files\\RandomForestClassifier_best_param.pkl;data_files',
    '--add-data=data_files\\KNeighborsClassifier.pkl;data_files',
    '--add-data=data_files\\KNeighborsClassifier_best_param.pkl;data_files',
    '--hidden-import=joblib',
    '--collect-submodules=sklearn',
    
])