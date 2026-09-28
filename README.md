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

The converter reads the native Dicom2H5 contract (`mag`, `flow`, `RR`,
`Resolution`, `VENC`, and `VENCOrder`). Native `flow` velocity channels are
converted with `velocity / VENC * pi` before DICOM pixel encoding. If the input
flow is a symmetric normalized signal, pass `--norm pi` (or a numeric scale);
the conversion then uses `flow / norm * VENC`. For an H5 file containing
multiple sequence groups, the command exports all groups by default; select
one explicitly with `--source-group <group>`. PCMRA magnitude generation is
enabled by default; use `--no-pcmra` to keep the native magnitude. The four
Siemens shell DICOMs are bundled inside this library under
`src/h52dicom/templates/dicom_shells`. Output metadata is copied from those
shells without runtime anonymization. To anonymize output, replace the bundled
shells with audited anonymized shells before running the converter.

The H5 `corr` phase cache is ignored by default. Enable it explicitly with
`--corr`; this mode requires real-valued input flow phases in `[-pi, pi]` and
subtracts the cached correction with phase wrapping. Native velocity H5 data
in cm/s should therefore use the default (`corr` disabled).

Legacy entry point:

```bash
python H52Dicom.py --file-path <input.h5> --out-path <output_dir>
```

The shell DICOM root is resolved relative to this installed library.
