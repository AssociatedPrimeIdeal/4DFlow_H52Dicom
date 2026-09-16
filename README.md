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

To remove patient/scanner metadata inherited from the Siemens template while
keeping the generated patient name/ID, dates, ShanghaiTech labels, and CVI
flow tags, add `--anonymize`:

```bash
h52dicom --file-path <input.h5> --out-path <output_dir> --patient-name <case> --anonymize
```

The anonymization flag is opt-in; without it, template metadata behavior is
unchanged.

Legacy entry point:

```bash
python H52Dicom.py --file-path <input.h5> --out-path <output_dir>
```

The template DICOM root is hard-coded in `src/h52dicom/converter.py` for your own use.
