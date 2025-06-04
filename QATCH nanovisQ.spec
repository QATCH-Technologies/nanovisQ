# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import exec_statement

best_fw_version = exec_statement(f"import os; os.chdir(r'{SPECPATH}'); from QATCH.core.constants import Constants; print(Constants.best_fw_version)")
print(f"Embedded firmware version: '{best_fw_version}'")

block_cipher = None

data_files = [
    ( "docs", "docs" ),
    ( "QATCH\\.libs", "."),
    ( "QATCH\\*.pdf", "QATCH" ),
    ( "QATCH\\icons", "QATCH\\icons" ),
    ( "QATCH\\models", "QATCH\\models" ),
    ( "QATCH\\QModel\\SavedModels", "QATCH\\QModel\\SavedModels" ),
    ( "QATCH\\resources", "QATCH\\resources" ),
    ( "QATCH\\VisQAI\\assets", "QATCH\\VisQAI\\assets" ),
    ( f"QATCH_Q-1_FW_py_{best_fw_version}\\*.hex", f"QATCH_Q-1_FW_py_{best_fw_version}" ),
    ( f"QATCH_Q-1_FW_py_{best_fw_version}\\*.pdf", f"QATCH_Q-1_FW_py_{best_fw_version}" ),
    ( "tools", "tools" )
]
a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[ # these may be required if the PyInstaller hooks for these modules get removed or change unexpectedly
        # ( "C:\\Users\\Alexander J. Ross\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\tensorflow\\python\\_pywrap_tensorflow_internal.pyd", "." ),
        # ( "C:\\Users\\Alexander J. Ross\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\xgboost\\lib\\xgboost.dll", "xgboost\\lib" ),
        # ( "C:\\Users\\Alexander J. Ross\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\xgboost\\VERSION", "xgboost" )
    ],
    datas=data_files,
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
splash = Splash(
    'QATCH\\icons\\qatch-splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(10,470),
    text_size=10,
    minify_script=True,
    always_on_top=False,
)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries, # onefile
    a.zipfiles, # onefile
    a.datas, # onefile
    splash, # splash (onedir and/or onefile)
    splash.binaries, # splash onefile
    [],
    exclude_binaries=False,
    name='QATCH nanovisQ',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='version.rc',
    icon=['QATCH\\ui\\favicon.ico'],
)
# uncomment for onedir:
# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
# #    splash.binaries, # splash onedir
#     strip=False,
#     upx=False,
#     upx_exclude=[],
#     name='QATCH nanovisQ',
# )
