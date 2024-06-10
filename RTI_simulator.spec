# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['joblib']
hiddenimports += collect_submodules('sklearn')


a = Analysis(
    ['RTI_simulator.py'],
    pathex=[],
    binaries=[],
    datas=[('data_files\\Sub10-1__result.mat', 'data_files'), ('data_files\\Sub10-2__result.mat', 'data_files'), ('data_files\\Sub1-1__result.mat', 'data_files'), ('data_files\\Sub11-1__result.mat', 'data_files'), ('data_files\\Sub11-2__result.mat', 'data_files'), ('data_files\\Sub12-1__result.mat', 'data_files'), ('data_files\\Sub12-2__result.mat', 'data_files'), ('data_files\\Sub13-1__result.mat', 'data_files'), ('data_files\\Sub13-2__result.mat', 'data_files'), ('data_files\\Sub14-1__result.mat', 'data_files'), ('data_files\\Sub14-2__result.mat', 'data_files'), ('data_files\\Sub3-1__result.mat', 'data_files'), ('data_files\\Sub3-2__result.mat', 'data_files'), ('data_files\\Sub4-1__result.mat', 'data_files'), ('data_files\\Sub4-2__result.mat', 'data_files'), ('data_files\\Sub5-1__result.mat', 'data_files'), ('data_files\\Sub5-2__result.mat', 'data_files'), ('data_files\\Sub6-1__result.mat', 'data_files'), ('data_files\\Sub7-1__result.mat', 'data_files'), ('data_files\\Sub8-1__result.mat', 'data_files'), ('data_files\\Sub8-2__result.mat', 'data_files'), ('data_files\\Sub9-1__result.mat', 'data_files'), ('data_files\\Sub9-2__result.mat', 'data_files'), ('data_files\\RandomForestClassifier.pkl', 'data_files'), ('data_files\\RandomForestClassifier_best_param.pkl', 'data_files'), ('data_files\\KNeighborsClassifier.pkl', 'data_files'), ('data_files\\KNeighborsClassifier_best_param.pkl', 'data_files')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [('O', None, 'OPTION')],
    name='RTI_simulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.jpg'],
    hide_console='hide-late',
)
