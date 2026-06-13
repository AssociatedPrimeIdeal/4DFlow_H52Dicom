# 4DFlow_H52Dicom

Convert 4D flow H5/MAT arrays to DICOM series.

## Install

```bash
pip install .
```

## Usage

```bash
h52dicom --file-path <input.h5> --out-path <output_dir>
```

Legacy entry point:

```bash
python H52Dicom.py --file-path <input.h5> --out-path <output_dir>
```

The template DICOM root is hard-coded in `src/h52dicom/converter.py` for your own use.
