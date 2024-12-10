# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files
import os

from PyInstaller.utils.hooks import collect_all


a = Analysis(
    ['main.py'],
    pathex=[],  # Thư mục làm việc
    binaries=[],
    datas=[
        # Thêm các thư mục và tệp vào bundle
        ('app/models.py', 'app/models.py'),
        ('app/routes.py', 'app/routes.py'),
        ('app/utils.py', 'app/utils.py'),
        ('app/__init__.py', 'app/__init__.py'),
        ('test.py', 'test.py'),
        ('main.py', 'main.py'),
        ('requirements.txt', 'requirements.txt'),
        ('main.spec', 'main.spec'),
        
        ('config/settings.py', 'config/settings.py'),
        
        
    ],
    hiddenimports=[
        'tensorflow',  # TensorFlow
        'sklearn',     # Scikit-learn
        'flask',       # Flask
        'pyodbc',      # PyODBC
        'itertools',   # itertools (nếu cần)
        'app.models',  # app.models để bao gồm các mô-đun trong ứng dụng của bạn
        'app.utils',   # app.utils để bao gồm các mô-đun trong ứng dụng của bạn
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Tạo gói EXE
pyz = PYZ(a.pure)

# Đóng gói và tạo EXE
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
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
)
