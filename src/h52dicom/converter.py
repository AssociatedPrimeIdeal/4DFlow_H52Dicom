"""Convert 4D flow H5 arrays to DICOM series."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from tqdm import tqdm

DEFAULT_TARGET_FOLDERS = (
    "series0096-Body",
    "series0093-Body",
    "series0094-Body",
    "series0095-Body",
)
DEFAULT_SERIES_DESCRIPTIONS = (
    "785B_4Dflow_ePAT3_retro_native",
    "785B_4Dflow_ePAT3_retro_native_P",
    "785B_4Dflow_ePAT3_retro_native_P",
    "785B_4Dflow_ePAT3_retro_native_P",
)
DEFAULT_PHASE_FLAGS = (False, True, True, True)
DEFAULT_DICOM_INDEX = (3, 0, 1, 2)
DEFAULT_TARGET_MAX = (540, 4095, 4095, 4095)
DEFAULT_TARGET_MIN = (0, 0, 0, 0)
DEFAULT_PROTOCOL_NAME = "785B_4Dflow_ePAT3_retro_native"
DEFAULT_TEMPLATE_ROOT = Path(
    "/nas-data/ryy_rawdata/zhupiter/1FlowReconEverything/GIT/4DFlowSOP/DICOM_TEMP/Siemens"
)


def _current_date() -> str:
    return datetime.now().strftime("%Y%m%d")


def _load_array(file_path: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(file_path, (str, Path)):
        with h5py.File(file_path, "r") as handle:
            key = next(iter(handle.keys()))
            array = handle[key][:]
        return np.asarray(array)
    return np.asarray(file_path)


def _normalize_orientation_name(orientation: str | None) -> str:
    if orientation is None:
        return "Sag"
    orientation = orientation.strip().capitalize()
    if orientation not in {"Tra", "Cor", "Sag"}:
        raise ValueError(f"Unsupported orientation: {orientation}")
    return orientation


def _order_to_target(order: Sequence[str], target_order: Sequence[str]) -> tuple[list[int], np.ndarray]:
    order_map = {item: idx for idx, item in enumerate(order)}
    new_indices: list[int] = []
    signs: list[int] = []
    for target in target_order:
        if target in order_map:
            new_indices.append(order_map[target])
            signs.append(1)
            continue
        reverse = target[::-1]
        if reverse in order_map:
            new_indices.append(order_map[reverse])
            signs.append(-1)
            continue
        raise ValueError(f"Cannot map orientation {target!r} from {list(order)!r}")
    return new_indices, np.asarray(signs, dtype=np.int8)


def _build_labels(orientation: str, venc_value: Sequence[int]) -> tuple[list[str], list[str]]:
    v0, v1, v2 = [int(v) for v in venc_value]
    if orientation == "Tra":
        sequence_info = [
            "",
            f"v{v0}_through",
            f"v{v1}_inplane_ap",
            f"v{v2}_inplane_rl",
        ]
        sequence_name = [
            "WIP_fl3d1r2",
            f"WIP_f_v{v0}in",
            f"WIP_f_v{v1}ap",
            f"WIP_f_v{v2}rl",
        ]
    elif orientation == "Sag":
        sequence_info = [
            "",
            f"v{v0}_inplane_ap",
            f"v{v1}_inplane_rl",
            f"v{v2}_through",
        ]
        sequence_name = [
            "WIP_fl3d1r2",
            f"WIP_f_v{v0}ap",
            f"WIP_f_v{v1}rl",
            f"WIP_f_v{v2}in",
        ]
    else:
        sequence_info = [
            "",
            f"v{v0}_inplane_ap",
            f"v{v1}_through",
            f"v{v2}_inplane_rl",
        ]
        sequence_name = [
            "WIP_fl3d1r2",
            f"WIP_f_v{v0}ap",
            f"WIP_f_v{v1}in",
            f"WIP_f_v{v2}rl",
        ]
    return sequence_info, sequence_name


def _resolve_reference_files() -> list[Path]:
    root = DEFAULT_TEMPLATE_ROOT
    files = [root / folder / "img0001--37.8632.dcm" for folder in DEFAULT_TARGET_FOLDERS]
    for file in files:
        if not file.exists():
            raise FileNotFoundError(f"Reference DICOM file not found: {file}")
    return files


def _safe_scale_unit(x: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(x))
    if max_abs == 0:
        return np.zeros_like(x, dtype=np.float32)
    return (x + max_abs) / (2.0 * max_abs)


def _safe_minmax_unit(x: np.ndarray) -> np.ndarray:
    min_v = np.min(x)
    max_v = np.max(x)
    span = max_v - min_v
    if span == 0:
        return np.zeros_like(x, dtype=np.float32)
    return (x - min_v) / span


def _set_if_present(ds: pydicom.dataset.Dataset, tag: int, value) -> None:
    if tag in ds:
        ds[tag].value = value


def _write_series(
    data: np.ndarray,
    template_file: Path,
    output_dir: Path,
    para: dict,
    rr: float,
    phase: bool = True,
) -> list[Path]:
    ori_dcm = pydicom.dcmread(str(template_file))
    tar_dcm = deepcopy(ori_dcm)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_if_present(tar_dcm, 0x0020000D, para["study_uid"])
    _set_if_present(tar_dcm, 0x0020000E, para["series_uid"])
    _set_if_present(tar_dcm, 0x00200011, para["SeriesNumber"])
    _set_if_present(tar_dcm, 0x00080012, para["date"])
    _set_if_present(tar_dcm, 0x00080020, para["date"])
    _set_if_present(tar_dcm, 0x00080021, para["date"])
    _set_if_present(tar_dcm, 0x00080022, para["date"])
    _set_if_present(tar_dcm, 0x00080023, para["date"])
    _set_if_present(tar_dcm, 0x00080080, para["institution_name"])
    _set_if_present(tar_dcm, 0x00081030, para["study_description"])
    _set_if_present(tar_dcm, 0x0008103E, para["series_description"])
    _set_if_present(tar_dcm, 0x00100010, para["patient_name"])
    _set_if_present(tar_dcm, 0x00100020, f"{para['patient_name']}_ID")
    _set_if_present(tar_dcm, 0x00180024, para["sequence_name"])
    _set_if_present(tar_dcm, 0x00180050, para["slice_thickness"])
    _set_if_present(tar_dcm, 0x00180080, para["repetition_time"])
    _set_if_present(tar_dcm, 0x00180081, para["echo_time"])
    _set_if_present(tar_dcm, 0x00180089, para["phase_encoding_step"])
    _set_if_present(tar_dcm, 0x00180093, para["percent_sampling"])
    _set_if_present(tar_dcm, 0x00180094, para["percent_phase_fov"])
    _set_if_present(tar_dcm, 0x00181030, para["protocol_name"])
    _set_if_present(tar_dcm, 0x00181062, para["nominal_interval"])
    _set_if_present(tar_dcm, 0x00181090, para["cardiac_phase"])
    _set_if_present(tar_dcm, 0x00181310, para["acquisition_matrix"])
    _set_if_present(tar_dcm, 0x00191017, para["slice_resolution"])
    _set_if_present(tar_dcm, 0x00280010, para["rows"])
    _set_if_present(tar_dcm, 0x00280011, para["columns"])
    _set_if_present(tar_dcm, 0x00280030, para["pixel_spacing"])
    _set_if_present(tar_dcm, 0x00291009, para["date"])
    _set_if_present(tar_dcm, 0x00291019, para["date"])
    _set_if_present(tar_dcm, 0x00400244, para["date"])
    _set_if_present(tar_dcm, 0x00400254, para["study_description"])
    _set_if_present(tar_dcm, 0x0051100B, para["acq_matrix_text"])
    _set_if_present(tar_dcm, 0x0051100C, para["fov_text"])
    _set_if_present(tar_dcm, 0x00511017, para["slice_text"])
    if phase:
        _set_if_present(tar_dcm, 0x00511014, para["sequence_info"])

    content_time = deepcopy(ori_dcm.get(0x00080033)).value if ori_dcm.get(0x00080033) else "0"
    instance_time = deepcopy(ori_dcm.get(0x00080013)).value if ori_dcm.get(0x00080013) else "0"
    instance_uid = deepcopy(ori_dcm.get(0x00080018)).value if ori_dcm.get(0x00080018) else generate_uid()
    slice_position = deepcopy(ori_dcm.get(0x00191015)).value if ori_dcm.get(0x00191015) else [0, 0, 0]
    image_position = deepcopy(ori_dcm.get(0x00200032)).value if ori_dcm.get(0x00200032) else [0, 0, 0]

    point_time_add = 0.000421
    full_time_add = 70
    total_iterations = int(para["cardiac_phase"]) * data.shape[0]
    out_files: list[Path] = []
    index = 1

    with tqdm(total=total_iterations) as pbar:
        for i in range(int(para["cardiac_phase"])):
            point_time = 0.263421 + i * point_time_add
            full_time = 2634211212 * 100 + i * full_time_add

            _set_if_present(tar_dcm, 0x00181060, str(np.around(i * rr / int(para["cardiac_phase"]), decimals=3)))

            for j in range(data.shape[0]):
                content_time = str(np.round(float(content_time) + point_time, 6))
                instance_time = str(np.round(float(instance_time) + point_time, 6))
                instance_uid_str = str(instance_uid)
                uid_tail = str(int(instance_uid_str.split(".")[-1]) + full_time)
                instance_uid = ".".join(instance_uid_str.split(".")[:-1] + [uid_tail])

                _set_if_present(tar_dcm, 0x00080033, content_time)
                _set_if_present(tar_dcm, 0x00080013, instance_time)
                _set_if_present(tar_dcm, 0x00080018, instance_uid)
                _set_if_present(tar_dcm, 0x00191015, deepcopy(slice_position))
                _set_if_present(tar_dcm, 0x00200013, str(index))
                _set_if_present(tar_dcm, 0x00200032, deepcopy(image_position))
                _set_if_present(tar_dcm, 0x00201041, str(image_position[-1]))

                slice_data = np.asarray(np.squeeze(data[j, :, :, i]))
                _set_if_present(tar_dcm, 0x00280106, int(np.min(slice_data)))
                _set_if_present(tar_dcm, 0x00280107, int(np.max(slice_data)))
                _set_if_present(
                    tar_dcm,
                    0x0051100D,
                    f"SP {'H' if float(image_position[-1]) >= 0 else 'F'}{np.round(np.abs(float(image_position[-1])))}",
                )

                tar_dcm.PixelData = slice_data.tobytes()
                output_file = output_dir / f"img{index:04d}-{np.round(image_position[-1], 4)}.dcm"
                tar_dcm.save_as(str(output_file))
                out_files.append(output_file)

                slice_position[-1] = float(slice_position[-1]) + float(para["slice_resolution"])
                image_position[-1] = float(image_position[-1]) + float(para["slice_resolution"])
                index += 1
                pbar.update(1)

    return out_files


def convert_array_to_dicom(
    file_path: str | Path | np.ndarray,
    out_path: str | Path,
    orientation: str | None = None,
    venc_order: Sequence[str] = ("RL", "AP", "FH"),
    venc_value: Sequence[float] = (150, 150, 150),
    spatial_order: Sequence[str] = ("HF", "AP", "RL"),
    pixel_size: Sequence[float] = (1.8, 1.9, 2.4),
    rr: float = 1000,
    patient_name: str = "patient",
    date: str | None = None,
) -> list[Path]:
    orientation = _normalize_orientation_name(orientation)
    venc_order = np.asarray(venc_order)
    venc_value = np.asarray(venc_value, dtype=np.float32)
    spatial_order = np.asarray(spatial_order)
    pixel_size = np.asarray(pixel_size, dtype=np.float32)
    date = date or _current_date()

    if venc_order.size != 3 or spatial_order.size != 3 or venc_value.size != 3 or pixel_size.size != 3:
        raise ValueError("venc_order, venc_value, spatial_order, and pixel_size must each have 3 elements.")

    if orientation == "Tra":
        venc_order_target = ("HF", "AP", "RL")
        spatial_order_target = ("FH", "AP", "RL")
    elif orientation == "Cor":
        venc_order_target = ("HF", "AP", "RL")
        spatial_order_target = ("AP", "HF", "RL")
    else:
        venc_order_target = ("HF", "AP", "RL")
        spatial_order_target = ("RL", "HF", "AP")

    venc_indices, venc_signs = _order_to_target(venc_order, venc_order_target)
    spatial_indices, spatial_signs = _order_to_target(spatial_order, spatial_order_target)

    venc_value = venc_value[venc_indices]
    pixel_size = pixel_size[spatial_indices]

    img = _load_array(file_path)
    if img.ndim != 5:
        raise ValueError(f"Expected a 5D array, got shape {img.shape!r}.")
    if img.shape[-1] != 4:
        img = np.transpose(img, (4, 3, 2, 1, 0))

    mag = img[..., 0]
    max_mag = np.max(mag)
    if max_mag > 0:
        mag = mag / max_mag
    else:
        mag = np.zeros_like(mag, dtype=np.float32)

    flow = img[..., 1:]
    pcmra = np.sqrt(np.sum(flow**2, axis=-1)) * mag
    img = np.asarray(img, dtype=np.float32)
    img[..., 0] = pcmra
    img[..., 1:] = flow

    img = img[..., list(np.asarray(venc_indices) + 1) + [0]].transpose(
        spatial_indices[0], spatial_indices[1], spatial_indices[2], 4, 3
    )
    img[..., :-1, :] *= venc_signs[None, None, None, :, None]
    img[..., :-1, :] = img[..., :-1, :] / np.pi * venc_value[None, None, None, :, None]

    actual_venc_value = np.ceil(np.max(np.abs(img[..., :-1, :]), axis=(0, 1, 2, 4))).astype("int16")
    if spatial_signs[0] == -1:
        img = img[::-1]
    if spatial_signs[1] == -1:
        img = img[:, ::-1]
    if spatial_signs[2] == -1:
        img = img[:, :, ::-1]

    sequence_info, sequence_name = _build_labels(orientation, actual_venc_value)
    reference_paths = _resolve_reference_files()

    target_root = Path(out_path)
    target_root.mkdir(parents=True, exist_ok=True)

    dicom_index = DEFAULT_DICOM_INDEX
    target_min = DEFAULT_TARGET_MIN
    target_max = DEFAULT_TARGET_MAX

    spe, pe, fe, nv, nt = img.shape
    if nv != 4:
        raise ValueError(f"Expected 4 channels after reordering, got {nv}.")
    fov = np.array([spe * pixel_size[0], pe * pixel_size[1], fe * pixel_size[2]], dtype=np.float32)
    matrix_size = np.array([spe, pe, fe], dtype=np.int32)
    mean_venc = int(np.mean(actual_venc_value))
    series_number_base = 300 + mean_venc
    out_files: list[Path] = []
    study_uid = generate_uid()

    for target_index in range(nv):
        para = {
            "date": date,
            "institution_name": "ShanghaiTech",
            "study_description": "ShanghaiTech",
            "series_description": DEFAULT_SERIES_DESCRIPTIONS[target_index],
            "SeriesNumber": series_number_base + dicom_index[target_index],
            "study_uid": study_uid,
            "series_uid": generate_uid(),
            "patient_name": patient_name,
            "sequence_name": sequence_name[target_index],
            "slice_thickness": str(fov[0] / matrix_size[0]),
            "repetition_time": "6",
            "echo_time": "3.1",
            "phase_encoding_step": str(matrix_size[1]),
            "percent_sampling": "100.0",
            "percent_phase_fov": "100.0",
            "protocol_name": DEFAULT_PROTOCOL_NAME,
            "nominal_interval": str(int(np.ceil(rr))),
            "cardiac_phase": str(nt),
            "acquisition_matrix": [int(matrix_size[2]), 0, 0, int(matrix_size[1])],
            "slice_resolution": str(pixel_size[0]),
            "rows": int(matrix_size[1]),
            "columns": int(matrix_size[2]),
            "fov_text": f"FoV {fov[1]}*{fov[2]}",
            "slice_text": f"SL {pixel_size[0]}i",
            "acq_matrix_text": f"{matrix_size[1]}*{matrix_size[2]}",
            "pixel_spacing": [float(pixel_size[2]), float(pixel_size[1])],
            "sequence_info": sequence_info[target_index],
            "ori_min": int(np.min(img[..., :-1, :])),
            "ori_max": int(np.max(img[..., :-1, :])),
            "target_min": target_min[target_index],
            "target_max": target_max[target_index],
        }

        imgv = np.squeeze(img[..., dicom_index[target_index], :])
        if dicom_index[target_index] == 3:
            imgv = _safe_minmax_unit(imgv)
        else:
            imgv = _safe_scale_unit(imgv)
        imgv = (imgv * (para["target_max"] - para["target_min"])) + para["target_min"]
        imgv = np.round(imgv).astype("uint16")

        target_dir = target_root / DEFAULT_TARGET_FOLDERS[target_index]
        out_files.extend(
            _write_series(
                imgv,
                reference_paths[target_index],
                target_dir,
                para,
                rr,
                DEFAULT_PHASE_FLAGS[target_index],
            )
        )

    return out_files


def array2dicom(*args, **kwargs):
    return convert_array_to_dicom(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert 4D flow H5 arrays to DICOM series.")
    parser.add_argument("--file-path", type=str, required=True, help="Input H5/MAT file or array-backed file.")
    parser.add_argument("--out-path", type=str, required=True, help="Output directory.")
    parser.add_argument("--orientation", type=str, default="Sag", help="Tra, Cor, or Sag.")
    parser.add_argument("--venc-order", type=str, nargs=3, default=["RL", "AP", "FH"], help="Velocity encoding order.")
    parser.add_argument("--venc-value", type=float, nargs=3, default=[150, 150, 150], help="Velocity encoding values.")
    parser.add_argument("--spatial-order", type=str, nargs=3, default=["HF", "AP", "RL"], help="Spatial order.")
    parser.add_argument("--pixel-size", type=float, nargs=3, default=[1.8, 1.9, 2.4], help="Pixel size.")
    parser.add_argument("--rr", type=float, default=1000, help="RR interval.")
    parser.add_argument("--patient-name", type=str, default="patient", help="Patient name.")
    parser.add_argument("--date", type=str, default=None, help="Study date (YYYYMMDD).")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    convert_array_to_dicom(
        file_path=args.file_path,
        out_path=args.out_path,
        orientation=args.orientation,
        venc_order=args.venc_order,
        venc_value=args.venc_value,
        spatial_order=args.spatial_order,
        pixel_size=args.pixel_size,
        rr=args.rr,
        patient_name=args.patient_name,
        date=args.date,
    )
    return 0
