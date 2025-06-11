# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import exec_statement
import logging

best_fw_version = exec_statement(f"import os; os.chdir(r'{SPECPATH}'); from QATCH.core.constants import Constants; print(Constants.best_fw_version)")
logging.info(f"Embedded firmware version: '{best_fw_version}'")

# Warning: Using an AES block cipher can trigger false positives with AV software
block_cipher = None
data_files = [
    ( "docs", "docs" ),
    ( "QATCH\\.libs", "."),
    ( "QATCH\\*.pdf", "QATCH" ),
    ( "QATCH\\icons", "QATCH\\icons" ),
    ( "QATCH\\models", "QATCH\\models" ),
    ( "QATCH\\QModel\\SavedModels", "QATCH\\QModel\\SavedModels" ),
    ( "QATCH\\resources", "QATCH\\resources" ),
    ( f"QATCH_Q-1_FW_py_{best_fw_version}\\*.hex", f"QATCH_Q-1_FW_py_{best_fw_version}" ),
    ( f"QATCH_Q-1_FW_py_{best_fw_version}\\*.pdf", f"QATCH_Q-1_FW_py_{best_fw_version}" ),
    ( "tools", "tools" )
]
# NOTE: Use of UPX requires `upx.exe` to be in the `.venv\Scripts` folder (not there by default)
upx_exclude = [
# Python 3.x base library DLLs
    'python3.dll', 
    'vcruntime140.dll', 
    'vcruntime140_1.dll',

# PyQt5 Core DLLs and plugins (GUI libraries often fail when compressed)
    'Qt5Core.dll',
    'Qt5Gui.dll',
    'Qt5Widgets.dll',
    'Qt5Network.dll',
    'Qt5PrintSupport.dll',
    'Qt5Qml.dll',
    'Qt5Quick.dll',
    'Qt5Svg.dll',
    'Qt5Multimedia.dll',
    'PyQt5',

# TensorFlow and dependencies (very sensitive to compression)
    'tensorflow',
    'tensorflow_intel',
    'tensorflow_io_gcs_filesystem',
    'libtensorflow_framework.dll',
    'libtensorflow.dll',
    'libtensorflow_cc.dll',

# gRPC and protobuf often include native binaries
    'grpcio',
    'protobuf',

# NumPy and SciPy contain compiled .pyd/.dll extensions
    'numpy',
    'scipy',

# XGBoost includes C++ binaries
    'xgboost',
    'xgboost.dll',

# Cryptography — may rely on native libcrypto
    'pycryptodomex',
    'libcrypto-*.dll',
    'libssl-*.dll',

# Matplotlib and Qt bindings
    'matplotlib',
    'pyqtgraph',

# HDF5 library bindings
    'h5py',

# Anything compiled in C/C++ via PyPI wheels
    'hid',  # USB HID wrapper
    'simdkalman',
]
# The Analysis block is required for both `--onefile` and `--onedir` modes
a = Analysis(
    ['app.py'],
    pathex=[],
# These binaries are required in-lieu of custom PyInstaller hooks (which are not portable or stable)
    binaries=[
        ( ".venv\\Lib\\site-packages\\py4j\\*.py", "py4j" ),
        ( ".venv\\Lib\\site-packages\\xgboost\\lib\\xgboost.dll", "xgboost\\lib" ),
        ( ".venv\\Lib\\site-packages\\xgboost\\VERSION", "xgboost" )
    ],
    datas=data_files,
# Hidden imports may be required for PyInstaller to identify deeply nested or runtime-loaded modules
    hiddenimports=['charset_normalizer.md__mypyc', 'numpy.core.multiarray'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['QATCH.nightly'], # ['pyqtgraph.opengl'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
# The `splash` (experimental) feature only really makes sense when using `--onefile` mode
# splash = Splash(
#     'QATCH\\icons\\qatch-splash.png',
#     binaries=a.binaries,
#     datas=[( "QATCH\\icons\\qatch-splash.png", "QATCH\\icons\\qatch-splash.png" )],
#     text_pos=(10,470),
#     text_size=10,
#     minify_script=True,
#     always_on_top=False,
# )
# The EXE block is required for both `--onefile` and `--onedir` modes
exe = EXE(
    pyz,
    a.scripts,
#    a.binaries, # onefile
#    a.zipfiles, # onefile
#    a.datas, # onefile
#    splash, # splash onefile
#    splash.binaries, # splash onefile
    [],
    exclude_binaries=True, # False,
    name='QATCH nanovisQ',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=upx_exclude,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version.rc',
    icon=['QATCH\\ui\\favicon.ico'],
)
# Collect is only required for `--onedir` compile mode
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=upx_exclude,
    name='QATCH nanovisQ',
)
