# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['backend/cli.py'],
    pathex=[],
    binaries=[],
    datas=[('backend', 'backend')],
    hiddenimports=['backend.organizer', 'backend.enhanced_cache', 'backend.utils', 'backend.models', 'backend.ai_providers', 'backend.config'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='smart-file-organizer-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='smart-file-organizer-backend',
)
