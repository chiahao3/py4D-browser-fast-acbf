# py4D-browser-fast-acbf

`py4D-browser-fast-acbf` wires
[fast-acbf](https://github.com/chiahao3/fast-acbf) into
[py4D-browser](https://github.com/sezelt/py4D-browser) as a plugin for
GPU-accelerated tcBF and acBF reconstruction. It uses CUDA when available,
then MPS on Apple Silicon, and falls back to CPU.

## Installation

Install into the same environment that runs py4D-browser:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate py4dgui
pip install -e /path/to/fast-acbf
pip install -e /path/to/py4D-browser-fast-acbf
```

## Usage

Start py4D-browser:

```bash
py4dgui
```

The plugin appears under **Plugins > fast-acbf** with:

- **Interactive Dashboard**: opens the live dashboard for reconstruction,
  optics/orientation overrides, preview images, and refinement history.
- **Quick Run (Last Config)**: runs the last saved configuration directly on
  the current datacube.
- **Configuration**: edits fast-acbf physics, device, reconstruction, output,
  orientation, aberration, and refinement settings.

By default the plugin reads py4D-browser calibration for scan step, reciprocal
pixel size, and accelerating voltage. If a circular diffraction detector ROI is
active, it can also infer the collection semiangle from that ROI. Otherwise,
set `max alpha`, `scan step`, `dk`, and voltage/wavelength manually in
Configuration.

Reconstruction results are sent back to py4D-browser as either the virtual image
or result image, depending on the configured output target.

## License

GNU GPLv3
