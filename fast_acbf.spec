"""
PyInstaller spec file for building a standalone macOS .app

    pyinstaller --clean fast_acbf.spec

Outputs: dist/py4DGUI.app
"""
import os
import sys
import sysconfig
import torch

# Ensure the fast-acbf plugin is on PyInstaller's import search path so it is
# collected as a submodule of the py4d_browser_plugin namespace package.
PLUGIN_SRC = os.path.join(os.getcwd(), "src")

block_cipher = None

# Include torch's MPS runtime library. Metal.framework and MetalKit.framework
# are system frameworks — dyld finds them at runtime, no bundling needed.
torch_base = os.path.dirname(torch.__file__)
mps_lib = os.path.join(torch_base, "lib", "mps_lib")

# Conda's non-framework Python on macOS doesn't get auto-detected by
# PyInstaller. Explicitly bundle the shared library so the bootloader
# can find it at runtime.
_python_libdir = sysconfig.get_config_var("LIBDIR")
_python_dylib = os.path.join(_python_libdir, "libpython3.12.dylib")

extra_binaries: list[tuple[str, str]] = []
if os.path.exists(mps_lib):
    extra_binaries.append((mps_lib, "torch/lib/"))
if os.path.exists(_python_dylib):
    extra_binaries.insert(0, (_python_dylib, "."))
else:
    raise FileNotFoundError(
        f"Could not find libpython dylib at expected path: {_python_dylib}"
    )

a = Analysis(
    ["launcher.py"],
    pathex=[PLUGIN_SRC],
    binaries=extra_binaries,
    datas=[],
    hiddenimports=[
        # Namespace package and our plugin module — makes pkgutil.iter_modules
        # discover fast_acbf inside the frozen bundle.
        "py4d_browser_plugin",
        "py4d_browser_plugin.fast_acbf",
        "py4d_browser_plugin.fast_acbf.__init__",
        "py4d_browser_plugin.fast_acbf.plugin",
        "py4d_browser_plugin.fast_acbf.config",
        "py4d_browser_plugin.fast_acbf.calibration",
        "py4d_browser_plugin.fast_acbf.utils",
        "py4d_browser_plugin.fast_acbf.worker",
        "py4d_browser_plugin.fast_acbf.solver_job",
        "py4d_browser_plugin.fast_acbf.lite_dock",
        "py4d_browser_plugin.fast_acbf.lite_runner",
        "py4d_browser_plugin.fast_acbf.dialogs",
        "py4d_browser_plugin.fast_acbf.dialogs.config_dialog",
        "py4d_browser_plugin.fast_acbf.dialogs.dashboard",
        "py4d_browser_plugin.fast_acbf.dialogs.lite_dialogs",
        "py4d_browser_plugin.fast_acbf.dialogs._widgets",
        "py4d_browser_plugin.fast_acbf.live_view",
        "py4d_browser_plugin.fast_acbf.live_view.solver",
        "py4d_browser_plugin.fast_acbf.live_view.engine",
        "py4d_browser_plugin.fast_acbf.live_view.metadata",
        "py4d_browser_plugin.fast_acbf.live_view.output",
        "py4d_browser_plugin.fast_acbf.live_view.dock",
        "py4d_browser_plugin.fast_acbf.live_view.session",
        "py4d_browser_plugin.fast_acbf.live_view.worker",

        # Core library
        "fast_acbf",

        # Torch MPS backend (macOS GPU acceleration via Metal)
        "torch.mps",

        # py4D-browser built-in plugins (namespace package neighbours)
        "py4d_browser_plugin.calibration_plugin",
        # "py4d_browser_plugin.logging_config_plugin",
        "py4d_browser_plugin.metadata_plugin",
        # "py4d_browser_plugin.tcBF_plugin",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "setuptools",
        "pip",
    ],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    onefile=False,
    exclude_binaries=False,
    name="py4DGUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=["*.dylib"],
    runtime_tmpdir=None,
    console=False,
    icon="py4DGUI.icns",
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name="py4DGUI.app",
    icon="py4DGUI.icns",
    bundle_identifier="com.py4d-browser.fast-acbf",
    info_plist={
        "CFBundleShortVersionString": "0.4.0",
        "LSMinimumSystemVersion": "13.0",
    },
    argv_emulation=False,
)
