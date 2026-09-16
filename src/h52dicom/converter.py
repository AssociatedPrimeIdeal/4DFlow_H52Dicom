"""Convert 4D flow H5 arrays to DICOM series."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from scipy.ndimage import uniform_filter
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
DEFAULT_ORIENTATION = "auto"

# Each value is (slice, image-row, image-column).  The image directions are
# intentionally right-handed: ``column_direction x row_direction`` equals the
# direction in which the emitted slices advance.  The previous sagittal tuple
# ended in AP while its ImageOrientationPatient described PA, so its pixels and
# geometry disagreed even though the private orientation tag said "Sag".
ORIENTATION_SPATIAL_ORDERS = {
    "Tra": ("FH", "AP", "RL"),
    "Cor": ("AP", "HF", "RL"),
    "Sag": ("RL", "HF", "PA"),
}

# DICOM patient coordinates are LPS: +X=left, +Y=posterior, +Z=head.
_AXIS_DIRECTION_LPS = {
    "RL": (1.0, 0.0, 0.0),
    "LR": (-1.0, 0.0, 0.0),
    "AP": (0.0, 1.0, 0.0),
    "PA": (0.0, -1.0, 0.0),
    "FH": (0.0, 0.0, 1.0),
    "HF": (0.0, 0.0, -1.0),
}
DEFAULT_TEMPLATE_ROOT = Path(
    "/nas-data/ryy_rawdata"
)

# Private fields needed by the project's Siemens flow classifier.  The
# template's other private blocks (especially 0019/0029 CSA) can contain
# patient weight, protocol history, scanner IDs and large Phoenix blobs.
ANON_KEEP_PRIVATE_TAGS = {
    0x00510010, 0x00511008, 0x00511009, 0x0051100A, 0x0051100B,
    0x0051100C, 0x0051100D, 0x0051100E, 0x00511013, 0x00511014,
    0x00511016, 0x00511017, 0x00511018, 0x00511019,
}
ANON_CLEAR_KEYWORDS = {
    "IssuerOfPatientID", "OtherPatientIDs", "OtherPatientNames",
    "PatientBirthDate", "PatientBirthTime", "PatientSex", "PatientAge",
    "PatientSize", "PatientWeight", "EthnicGroup", "Occupation",
    "AdditionalPatientHistory", "PatientComments", "MedicalRecordLocator",
    "AdmittingDiagnosesDescription", "ReferringPhysicianName",
    "ReferringPhysicianIdentificationSequence", "PerformingPhysicianName",
    "PerformingPhysicianIdentificationSequence", "NameOfPhysiciansReadingStudy",
    "PhysiciansOfRecord", "PhysiciansOfRecordIdentificationSequence",
    "OperatorsName", "OperatorIdentificationSequence", "InstitutionAddress",
    "InstitutionName", "InstitutionalDepartmentName", "StationName",
    "DeviceSerialNumber", "AccessionNumber", "StudyComments", "ImageComments",
    "PerformedProcedureStepID", "StudyTime", "SeriesTime", "AcquisitionTime",
    "ContentTime", "InstanceCreationTime",
}
ANON_DROP_SEQUENCE_KEYWORDS = {
    "ReferencedImageSequence", "ReferencedInstanceSequence",
    "ReferencedStudySequence", "ReferencedPatientSequence",
}


def _anonymize_template_dataset(dataset: pydicom.dataset.Dataset) -> None:
    """Remove template-derived identifiers while retaining Siemens flow tags."""
    clear_tags = {
        int(pydicom.datadict.tag_for_keyword(keyword))
        for keyword in ANON_CLEAR_KEYWORDS
        if pydicom.datadict.tag_for_keyword(keyword) is not None
    }

    def scrub_nested(item: pydicom.dataset.Dataset) -> None:
        for tag in list(item.keys()):
            element = item[tag]
            if element.VR == "SQ":
                if element.keyword in ANON_DROP_SEQUENCE_KEYWORDS:
                    del item[tag]
                    continue
                for child in element.value:
                    scrub_nested(child)
            if int(tag) in clear_tags:
                del item[tag]

    scrub_nested(dataset)
    kept_private = {
        int(tag): deepcopy(dataset[tag])
        for tag in list(dataset.keys())
        if tag.is_private and int(tag) in ANON_KEEP_PRIVATE_TAGS
    }
    dataset.remove_private_tags()
    for tag in sorted(kept_private):
        dataset.add(kept_private[tag])

    # This is a file-meta workstation identifier, not a flow parameter.
    if "SourceApplicationEntityTitle" in dataset.file_meta:
        dataset.file_meta.SourceApplicationEntityTitle = "ANON"
    dataset.file_meta.ImplementationClassUID = generate_uid()
    dataset.file_meta.ImplementationVersionName = "H52DICOM_ANON_1"


def _current_date() -> str:
    return datetime.now().strftime("%Y%m%d")


def _decode_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    values = np.asarray(value)
    if values.ndim == 0:
        values = values.reshape(1)
    output: list[str] = []
    for item in values.reshape(-1):
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        output.extend(token for token in str(item).replace(",", ";").split(";") if token)
    return [token.strip().upper() for token in output if token.strip()]


def _load_array(file_path: str | Path | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(file_path, (str, Path)):
        return np.asarray(file_path), {}

    with h5py.File(file_path, "r") as handle:
        key = "img_complex" if "img_complex" in handle else "img" if "img" in handle else next(iter(handle.keys()))
        array = np.asarray(handle[key][:])
        metadata: dict[str, Any] = {}
        for source_key, target_key in (
            ("VENC", "venc_value"),
            ("Venc", "venc_value"),
            ("Resolution", "pixel_size"),
            ("RR", "rr"),
            ("SpatialOrder", "spatial_order"),
            ("VENCOrder", "venc_order"),
            ("VencOrder", "venc_order"),
        ):
            if source_key in handle:
                metadata[target_key] = handle[source_key][()]
            elif source_key in handle.attrs:
                metadata[target_key] = handle.attrs[source_key]
        if "corr" in handle:
            metadata["corr"] = np.asarray(handle["corr"][:], dtype=np.float32)
    return array, metadata


def _normalize_channel_last(img: np.ndarray) -> np.ndarray:
    if img.ndim != 5:
        raise ValueError(f"Expected a 5D array, got shape {img.shape!r}.")
    if img.shape[-1] in (4, 7):
        return img
    if img.shape[0] in (4, 7):
        return np.transpose(img, (4, 3, 2, 1, 0))
    raise ValueError(f"Expected a 4- or 7-channel image array, got shape {img.shape!r}.")


def _normalize_corr_channel_last(corr: np.ndarray, img_shape: tuple[int, int, int, int, int]) -> np.ndarray:
    corr = np.asarray(corr, dtype=np.float32)
    columns, rows, slices, time_count, channels = img_shape
    expected_shapes = {
        (columns, rows, slices, 1, channels - 1),
        (columns, rows, slices, time_count, channels - 1),
    }
    if corr.shape in expected_shapes:
        return corr

    source_shapes = {
        (channels - 1, 1, slices, rows, columns),
        (channels - 1, time_count, slices, rows, columns),
    }
    if corr.shape in source_shapes:
        return corr.transpose(4, 3, 2, 1, 0)

    raise ValueError(
        f"corr shape {corr.shape} does not match image shape {img_shape}; "
        f"expected one of {sorted(expected_shapes | source_shapes)}"
    )


def _apply_phase_correction(img: np.ndarray, corr: np.ndarray | None) -> np.ndarray:
    if corr is None:
        return img
    if not np.iscomplexobj(img):
        raise ValueError("corr can only be applied to complex-valued phase encodings")
    corr = _normalize_corr_channel_last(corr, tuple(int(value) for value in img.shape))
    corrected = np.array(img, copy=True)
    corrected[..., 1:] *= np.exp(-1j * corr)
    return corrected


def _lap4(in_matrix: np.ndarray, direction: int, laplacian: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fftn(in_matrix))
    if direction == 1:
        spectrum *= laplacian
    elif direction == -1:
        denominator = np.where(laplacian == 0, 1, laplacian)
        spectrum /= denominator
    else:
        raise ValueError(f"Unsupported Laplacian direction: {direction}")
    return np.real(np.fft.ifftn(np.fft.ifftshift(spectrum)))


def _unwrap_lap4d(phase_wrapped: np.ndarray, temporal_scale: float = 2.0) -> np.ndarray:
    """Return the lap4D unwrapped phase used by the reference notebook."""
    if phase_wrapped.ndim != 4:
        raise ValueError(f"Expected 4D phase array, got {phase_wrapped.shape}")

    # unwrap_data adds the returned wrap count to the original phase, including
    # any odd boundary voxel excluded from the Laplacian crop.
    result = np.asarray(phase_wrapped, dtype=np.float32).copy()
    cropped = phase_wrapped[
        : phase_wrapped.shape[0] // 2 * 2,
        : phase_wrapped.shape[1] // 2 * 2,
        : phase_wrapped.shape[2] // 2 * 2,
        :,
    ]
    if 0 in cropped.shape:
        return result

    size_x, size_y, size_z, size_t = cropped.shape
    grid_x, grid_y, grid_z, grid_t = np.meshgrid(
        np.arange(-size_x // 2, size_x // 2),
        np.arange(-size_y // 2, size_y // 2),
        np.arange(-size_z // 2, size_z // 2),
        np.arange(-size_t // 2, size_t // 2),
        indexing="ij",
    )
    laplacian = (
        2.0 * np.cos(np.pi * grid_x / size_x)
        + 2.0 * np.cos(np.pi * grid_y / size_y)
        + 2.0 * np.cos(np.pi * grid_z / size_z)
        + temporal_scale * np.cos(np.pi * grid_t / size_t)
        - 6.0
        - temporal_scale
    )
    lap_wrapped = _lap4(cropped, 1, laplacian)
    lap_phase = (
        np.cos(cropped) * _lap4(np.sin(cropped), 1, laplacian)
        - np.sin(cropped) * _lap4(np.cos(cropped), 1, laplacian)
    )
    wraps = np.rint(_lap4(lap_phase - lap_wrapped, -1, laplacian) / (2.0 * np.pi)).astype(np.int8)
    result[:size_x, :size_y, :size_z, :] = cropped + 2.0 * np.pi * wraps
    return result


def _total_field_correction(phase_unwrapped: np.ndarray) -> np.ndarray:
    """Match utils_unwrap.total_field_correction with its all-ones mask."""
    energy = np.sum(phase_unwrapped, axis=(0, 1, 2))
    voxels_per_frame = phase_unwrapped.shape[0] * phase_unwrapped.shape[1] * phase_unwrapped.shape[2]
    if voxels_per_frame == 0:
        return phase_unwrapped
    unwrapped_energy = np.unwrap(
        energy,
        discont=np.pi * voxels_per_frame,
        period=2.0 * np.pi * voxels_per_frame,
    )
    wraps = np.rint((unwrapped_energy - energy) / (2.0 * np.pi * voxels_per_frame))
    return phase_unwrapped + wraps.reshape((1, 1, 1, -1)) * (2.0 * np.pi)


def _unwrap_high_venc_flow(phase_high: np.ndarray, venc_high: np.ndarray) -> np.ndarray:
    venc_high = np.asarray(venc_high, dtype=np.float32).reshape(3)
    flow_high = np.empty_like(phase_high, dtype=np.float32)
    for component in range(3):
        unwrapped_phase = _total_field_correction(_unwrap_lap4d(phase_high[..., component]))
        flow_high[..., component] = unwrapped_phase / np.pi * venc_high[component]
    return flow_high


def _dual_venc_flow(
    flow_low: np.ndarray,
    flow_high: np.ndarray,
    venc_low: np.ndarray,
    venc_high: np.ndarray,
) -> np.ndarray:
    """Fuse low/high VENC velocities using the notebook's dual-VENC procedure."""
    venc_low = np.asarray(venc_low, dtype=np.float32).reshape(3)
    venc_high = np.asarray(venc_high, dtype=np.float32).reshape(3)
    if flow_low.shape != flow_high.shape or flow_low.shape[-1] != 3:
        raise ValueError(f"Dual-VENC flow shape mismatch: low={flow_low.shape}, high={flow_high.shape}")

    # The reference pipeline first unwraps each high-VENC component in 4D and
    # applies its temporal field correction before low-VENC alias fusion.
    phase_high = flow_high / venc_high.reshape((1,) * (flow_high.ndim - 1) + (3,)) * np.pi
    flow_high = _unwrap_high_venc_flow(phase_high, venc_high)

    flow_dual = np.array(flow_low, dtype=np.float32, copy=True)
    for component in (2, 1, 0):
        low_venc = float(venc_low[component])
        high_venc = float(venc_high[component])
        low = flow_low[..., component]
        high = flow_high[..., component]

        # Pointwise candidate selection is equivalent to the notebook's Ju/K
        # calculation without materializing a candidate-by-volume array.
        sample_count = int(np.ceil(high_venc / low_venc)) + 2
        best_cost = np.full(low.shape, np.inf, dtype=np.float32)
        fused = np.zeros_like(low, dtype=np.float32)
        for multiplier in range(-sample_count, sample_count + 1):
            candidate = low + multiplier * low_venc
            cost = (1.0 - np.cos(np.pi / high_venc * (high - candidate))) + (
                1.0 - np.cos(np.pi / low_venc * (low - candidate))
            )
            cost = np.where(np.abs(candidate) > high_venc, np.inf, cost)
            selected = cost < best_cost
            fused[selected] = candidate[selected]
            best_cost[selected] = cost[selected]

        # Five rounds of spatial consistency and temporal discontinuity repair
        # match utils_dual.unwrap_dual(..., it=5).
        for _ in range(5):
            spatial_mean = uniform_filter(fused, size=(3, 3, 3, 1), mode="constant")
            temporal_difference = np.empty_like(fused)
            temporal_difference[..., 0] = fused[..., 0] - fused[..., -1]
            temporal_difference[..., 1:] = fused[..., 1:] - fused[..., :-1]
            discontinuity = np.abs(temporal_difference) > low_venc
            correction_cycles = np.rint((spatial_mean[discontinuity] - low[discontinuity]) / (2.0 * low_venc))
            fused[discontinuity] = low[discontinuity] + 2.0 * correction_cycles * low_venc

        flow_dual[..., component] = fused
    return flow_dual


def _complex_to_phase_layout(img: np.ndarray, venc_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    channels = int(img.shape[-1])
    reference = img[..., :1]
    magnitude = np.abs(reference[..., 0]).astype(np.float32)
    if channels == 4:
        if venc_value.size != 3:
            raise ValueError(f"Complex 4-channel data requires 3 VENC values, got {venc_value.tolist()}")
        phase = np.angle(img[..., 1:4] * np.conj(reference)).astype(np.float32)
        return np.concatenate([magnitude[..., None], phase], axis=-1), venc_value

    if venc_value.size != 6:
        raise ValueError(f"Complex 7-channel data requires 6 VENC values, got {venc_value.tolist()}")
    first, second = venc_value[:3], venc_value[3:]
    if np.all(first <= second) and np.any(first < second):
        low, high, low_channels, high_channels = first, second, img[..., 1:4], img[..., 4:7]
    elif np.all(second <= first) and np.any(second < first):
        low, high, low_channels, high_channels = second, first, img[..., 4:7], img[..., 1:4]
    else:
        raise ValueError(f"Cannot identify low/high VENC triplets from {venc_value.tolist()}")
    phase_low = np.angle(low_channels * np.conj(reference)).astype(np.float32)
    phase_high = np.angle(high_channels * np.conj(reference)).astype(np.float32)
    flow_low = phase_low / np.pi * low.reshape((1,) * (phase_low.ndim - 1) + (3,))
    flow_high = phase_high / np.pi * high.reshape((1,) * (phase_high.ndim - 1) + (3,))
    flow_dual = _dual_venc_flow(flow_low, flow_high, low, high)
    phase = flow_dual / high.reshape((1,) * (flow_dual.ndim - 1) + (3,)) * np.pi
    return np.concatenate([magnitude[..., None], phase.astype(np.float32)], axis=-1), high


def _normalize_orientation_name(orientation: str | None) -> str:
    if orientation is None:
        return DEFAULT_ORIENTATION
    normalized = orientation.strip().capitalize()
    if normalized.lower() == "auto":
        return "auto"
    if normalized not in ORIENTATION_SPATIAL_ORDERS:
        raise ValueError(f"Unsupported orientation: {orientation}")
    return normalized


def _axis_base(axis: str) -> str:
    axis = str(axis).upper()
    if axis in {"RL", "LR"}:
        return "RL"
    if axis in {"AP", "PA"}:
        return "AP"
    if axis in {"HF", "FH"}:
        return "HF"
    raise ValueError(f"Unsupported anatomical axis: {axis!r}")


def _resolve_orientation(
    orientation: str,
    spatial_shape: Sequence[int],
    spatial_order: Sequence[str],
) -> str:
    """Pick the plane whose physical slice axis has the fewest voxels."""
    if orientation != "auto":
        return orientation
    if len(spatial_shape) != 3 or len(spatial_order) != 3:
        raise ValueError("Cannot infer orientation without three spatial axes.")

    source_axis_sizes = {
        _axis_base(axis): int(size)
        for axis, size in zip(spatial_order, spatial_shape)
    }
    if set(source_axis_sizes) != {"RL", "AP", "HF"}:
        raise ValueError(f"Cannot infer orientation from spatial order {list(spatial_order)!r}")

    plane_slice_axis = {"Tra": "HF", "Cor": "AP", "Sag": "RL"}
    return min(
        plane_slice_axis,
        key=lambda plane: (source_axis_sizes[plane_slice_axis[plane]], ("Tra", "Cor", "Sag").index(plane)),
    )


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
        # Vendor Sag convention: RL is through-plane; the HF/AP encodings are
        # emitted in the FH/RL-labelled in-plane series slots.
        sequence_info = [
            "",
            f"v{v0}_inplane_fh",
            f"v{v1}_inplane_rl",
            f"v{v2}_through",
        ]
        sequence_name = [
            "WIP_fl3d1r2",
            f"WIP_f_v{v0}fh",
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


def _format_dicom_time(seconds: float) -> str:
    total_microseconds = int(round((float(seconds) % 86400.0) * 1_000_000)) % 86_400_000_000
    hours, remainder = divmod(total_microseconds, 3_600_000_000)
    minutes, remainder = divmod(remainder, 60_000_000)
    whole_seconds, microseconds = divmod(remainder, 1_000_000)
    return f"{hours:02d}{minutes:02d}{whole_seconds:02d}.{microseconds:06d}"


def _format_dicom_decimal(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"


def _format_slice_position_text(position: np.ndarray, orientation: str, normal: np.ndarray) -> str:
    """Format Siemens' private slice-position text in the selected plane."""
    positive_label = {"Tra": "H", "Cor": "P", "Sag": "L"}[orientation]
    negative_label = {"Tra": "F", "Cor": "A", "Sag": "R"}[orientation]
    location = float(np.dot(position, normal))
    label = positive_label if location >= 0 else negative_label
    return f"SP {label}{_format_dicom_decimal(abs(location))}"


def _set_if_present(ds: pydicom.dataset.Dataset, tag: int, value) -> None:
    if tag in ds:
        ds[tag].value = value


def _set_or_add(ds: pydicom.dataset.Dataset, tag: int, value) -> None:
    """Set a standard element, adding it when anonymization removed it."""
    if tag in ds:
        ds[tag].value = value
        return
    vr = pydicom.datadict.dictionary_VR(tag)
    ds.add_new(tag, vr, value)


def _orientation_geometry(orientation: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Derive DICOM geometry from the actual emitted (slice, row, column) axes."""
    try:
        slice_axis, row_axis, column_axis = ORIENTATION_SPATIAL_ORDERS[orientation]
    except KeyError as exc:
        raise ValueError(f"Unsupported orientation: {orientation}") from exc

    column_direction = np.asarray(_AXIS_DIRECTION_LPS[column_axis], dtype=np.float64)
    row_direction = np.asarray(_AXIS_DIRECTION_LPS[row_axis], dtype=np.float64)
    slice_direction = np.asarray(_AXIS_DIRECTION_LPS[slice_axis], dtype=np.float64)
    if not np.allclose(np.cross(column_direction, row_direction), slice_direction):
        raise AssertionError(f"DICOM axis convention is not right-handed for {orientation}")
    return tuple(column_direction.tolist() + row_direction.tolist()), tuple(slice_direction.tolist())


def _centered_image_origin(
    matrix_size: Sequence[int],
    pixel_size: Sequence[float],
    image_orientation: Sequence[float],
    slice_direction: Sequence[float],
) -> np.ndarray:
    """Return the first voxel center for a volume centered at patient origin."""
    slice_count, row_count, column_count = (int(value) for value in matrix_size)
    slice_spacing, row_spacing, column_spacing = (float(value) for value in pixel_size)
    column_direction = np.asarray(image_orientation[:3], dtype=np.float64)
    row_direction = np.asarray(image_orientation[3:], dtype=np.float64)
    slice_direction = np.asarray(slice_direction, dtype=np.float64)
    return -0.5 * (
        (slice_count - 1) * slice_spacing * slice_direction
        + (row_count - 1) * row_spacing * row_direction
        + (column_count - 1) * column_spacing * column_direction
    )


def _write_series(
    data: np.ndarray,
    template_file: Path,
    output_dir: Path,
    para: dict,
    rr: float,
    phase: bool = True,
    anonymize: bool = False,
) -> list[Path]:
    ori_dcm = pydicom.dcmread(str(template_file))
    tar_dcm = deepcopy(ori_dcm)
    if anonymize:
        _anonymize_template_dataset(tar_dcm)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CVI compatibility mode: pixels have already been transposed/flipped into
    # the requested view, but every orientation/position field remains in the
    # template's native transverse coordinate convention.  This intentionally
    # mirrors the historical Array2Dicom writer.

    _set_if_present(tar_dcm, 0x0020000D, para["study_uid"])
    _set_if_present(tar_dcm, 0x0020000E, para["series_uid"])
    _set_if_present(tar_dcm, 0x00200011, para["SeriesNumber"])
    _set_if_present(tar_dcm, 0x00080012, para["date"])
    _set_if_present(tar_dcm, 0x00080020, para["date"])
    _set_if_present(tar_dcm, 0x00080021, para["date"])
    _set_if_present(tar_dcm, 0x00080022, para["date"])
    _set_if_present(tar_dcm, 0x00080023, para["date"])
    # InstitutionName is cleared from the template first; this is the
    # generator-owned label explicitly allowed by the caller.
    _set_or_add(tar_dcm, 0x00080080, para["institution_name"])
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

    base_image_position = deepcopy(ori_dcm[0x00200032].value)
    private_position = ori_dcm.get(0x00191015)
    base_private_position = (
        deepcopy(private_position.value)
        if private_position is not None
        else deepcopy(base_image_position)
    )
    total_iterations = int(para["cardiac_phase"]) * data.shape[0]
    out_files: list[Path] = []
    index = 1

    with tqdm(total=total_iterations) as pbar:
        for i in range(int(para["cardiac_phase"])):
            cardiac_time_seconds = i * float(rr) / int(para["cardiac_phase"]) / 1000.0
            image_position = deepcopy(base_image_position)
            slice_position = deepcopy(base_private_position)

            _set_if_present(tar_dcm, 0x00181060, str(np.around(i * rr / int(para["cardiac_phase"]), decimals=3)))

            for j in range(data.shape[0]):
                instance_time = _format_dicom_time(12 * 3600 + cardiac_time_seconds + j * 0.001)

                _set_if_present(tar_dcm, 0x00080033, instance_time)
                _set_if_present(tar_dcm, 0x00080013, instance_time)
                instance_uid = generate_uid()
                _set_if_present(tar_dcm, 0x00080018, instance_uid)
                # Keep the Part 10 file-meta UID in sync with the dataset UID.
                # A mismatch makes the exported object formally invalid and some
                # DICOM importers reject it before inspecting the flow metadata.
                tar_dcm.file_meta.MediaStorageSOPInstanceUID = instance_uid
                tar_dcm.file_meta.MediaStorageSOPClassUID = tar_dcm.SOPClassUID
                _set_if_present(tar_dcm, 0x00191015, deepcopy(slice_position))
                _set_if_present(tar_dcm, 0x00200013, str(index))
                _set_if_present(tar_dcm, 0x00200032, deepcopy(image_position))
                slice_location = float(image_position[-1])
                _set_if_present(tar_dcm, 0x00201041, str(slice_location))

                slice_data = np.asarray(np.squeeze(data[j, :, :, i]))
                _set_if_present(tar_dcm, 0x00280106, int(np.min(slice_data)))
                _set_if_present(tar_dcm, 0x00280107, int(np.max(slice_data)))
                position_label = "H" if slice_location >= 0.0 else "F"
                _set_if_present(tar_dcm, 0x0051100D, f"SP {position_label}{np.round(abs(slice_location))}")

                tar_dcm.PixelData = slice_data.tobytes()
                output_file = output_dir / f"img{index:04d}-{np.round(slice_location, 4)}.dcm"
                tar_dcm.save_as(str(output_file))
                out_files.append(output_file)

                slice_position[-1] += float(para["slice_resolution"])
                image_position[-1] += float(para["slice_resolution"])
                index += 1
                pbar.update(1)

    return out_files


def convert_array_to_dicom(
    file_path: str | Path | np.ndarray,
    out_path: str | Path,
    orientation: str | None = DEFAULT_ORIENTATION,
    venc_order: Sequence[str] | None = None,
    venc_value: Sequence[float] | None = None,
    spatial_order: Sequence[str] | None = None,
    pixel_size: Sequence[float] | None = None,
    rr: float | None = None,
    patient_name: str | None = None,
    date: str | None = None,
    anonymize: bool = False,
) -> list[Path]:
    img, metadata = _load_array(file_path)
    if venc_order is None:
        venc_order = _decode_text_values(metadata.get("venc_order")) or ["RL", "AP", "FH"]
    if spatial_order is None:
        spatial_order = _decode_text_values(metadata.get("spatial_order")) or ["HF", "AP", "RL"]
    if venc_value is None:
        venc_value = metadata.get("venc_value", (150, 150, 150))
    if pixel_size is None:
        pixel_size = metadata.get("pixel_size", (1.8, 1.9, 2.4))
    if rr is None:
        rr = float(np.asarray(metadata.get("rr", 1000.0), dtype=float).reshape(-1)[0])
    if patient_name is None:
        patient_name = Path(file_path).stem if isinstance(file_path, (str, Path)) else "patient"

    venc_order = np.asarray([str(value).upper() for value in venc_order])
    venc_value = np.asarray(venc_value, dtype=np.float32).reshape(-1)
    spatial_order = np.asarray([str(value).upper() for value in spatial_order])
    # Keep double precision here because the historical Array2Dicom writer
    # builds its FoV text with Python ``str(float)``; converting to float32
    # first changes strings such as ``16.8`` into ``16.799999``.
    pixel_size = np.asarray(pixel_size, dtype=np.float64).reshape(-1)
    date = date or _current_date()

    if venc_order.size != 3 or spatial_order.size != 3 or pixel_size.size != 3:
        raise ValueError("venc_order, spatial_order, and pixel_size must each have 3 elements.")

    img = _normalize_channel_last(img)
    orientation = _resolve_orientation(
        _normalize_orientation_name(orientation),
        img.shape[:3],
        spatial_order,
    )
    venc_order_target = ("HF", "AP", "RL")
    spatial_order_target = ORIENTATION_SPATIAL_ORDERS[orientation]
    venc_indices, venc_signs = _order_to_target(venc_order, venc_order_target)
    spatial_indices, spatial_signs = _order_to_target(spatial_order, spatial_order_target)

    img = _apply_phase_correction(img, metadata.get("corr"))
    if np.iscomplexobj(img):
        img, venc_value = _complex_to_phase_layout(img, venc_value)
    elif img.shape[-1] != 4:
        raise ValueError(f"Real-valued input must have 4 channels, got shape {img.shape!r}.")
    if venc_value.size != 3:
        raise ValueError(f"Expected 3 effective VENC values, got {venc_value.tolist()}")
    venc_value = venc_value[venc_indices]
    pixel_size = pixel_size[spatial_indices]

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
    # H5 phase is encoded with the nominal/theoretical VENC.  Convert it to
    # velocity and clip values outside the acquisition range instead of
    # changing the VENC advertised by the DICOM tags.  CVI expects the pixel
    # encoding and ``vXXX_*`` labels to use the same fixed VENC.
    nominal_venc = np.asarray(venc_value, dtype=np.float32).copy()
    nominal_venc_broadcast = nominal_venc[None, None, None, :, None]
    velocity = img[..., :-1, :] / np.pi * nominal_venc_broadcast
    velocity = np.clip(velocity, -nominal_venc_broadcast, nominal_venc_broadcast)
    img[..., :-1, :] = velocity / nominal_venc_broadcast * np.pi

    actual_venc_value = np.ceil(nominal_venc).astype("int16")
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
    fov = np.array([spe * pixel_size[0], pe * pixel_size[1], fe * pixel_size[2]], dtype=np.float64)
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
            "slice_thickness": _format_dicom_decimal(fov[0] / matrix_size[0]),
            "repetition_time": "6",
            "echo_time": "3.1",
            "phase_encoding_step": str(matrix_size[1]),
            "percent_sampling": "100.0",
            "percent_phase_fov": "100.0",
            "protocol_name": DEFAULT_PROTOCOL_NAME,
            "nominal_interval": str(int(np.ceil(rr))),
            "cardiac_phase": str(nt),
            "acquisition_matrix": [int(matrix_size[2]), 0, 0, int(matrix_size[1])],
            "slice_resolution": _format_dicom_decimal(pixel_size[0]),
            "rows": int(matrix_size[1]),
            "columns": int(matrix_size[2]),
            # Match Array2Dicom4cry_v9 exactly (including its native float
            # representation, e.g. ``FoV 16.8*13.200000000000001``).
            "fov_text": f"FoV {fov[1]}*{fov[2]}",
            "slice_text": f"SL {_format_dicom_decimal(pixel_size[0])}i",
            "acq_matrix_text": f"{matrix_size[1]}*{matrix_size[2]}",
            "pixel_spacing": [_format_dicom_decimal(pixel_size[2]), _format_dicom_decimal(pixel_size[1])],
            "sequence_info": sequence_info[target_index],
            "orientation": orientation,
            "ori_min": int(np.min(img[..., :-1, :])),
            "ori_max": int(np.max(img[..., :-1, :])),
            "target_min": target_min[target_index],
            "target_max": target_max[target_index],
        }

        # Indexing the channel already yields [slice, row, column, cardiac].
        # Do not squeeze: a single-slice or single-phase acquisition must retain
        # the four axes expected by _write_series.
        imgv = img[..., dicom_index[target_index], :]
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
                anonymize=anonymize,
            )
        )

    return out_files


def array2dicom(*args, **kwargs):
    return convert_array_to_dicom(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert 4D flow H5 arrays to DICOM series.")
    parser.add_argument("--file-path", type=str, required=True, help="Input H5/MAT file or array-backed file.")
    parser.add_argument("--out-path", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--orientation",
        type=str,
        default=DEFAULT_ORIENTATION,
        choices=("auto", "Tra", "Cor", "Sag"),
        help="DICOM plane, or auto to use the smallest physical spatial axis as the slice axis.",
    )
    parser.add_argument("--venc-order", type=str, nargs=3, default=None, help="Velocity encoding order.")
    parser.add_argument("--venc-value", type=float, nargs="+", default=None, help="Velocity encoding values.")
    parser.add_argument("--spatial-order", type=str, nargs=3, default=None, help="Spatial order.")
    parser.add_argument("--pixel-size", type=float, nargs=3, default=None, help="Pixel size.")
    parser.add_argument("--rr", type=float, default=None, help="RR interval.")
    parser.add_argument("--patient-name", type=str, default=None, help="Patient name.")
    parser.add_argument("--date", type=str, default=None, help="Study date (YYYYMMDD).")
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Remove template-derived patient/scanner/CSA metadata while preserving generated name, ID, dates and CVI flow tags.",
    )
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
        anonymize=args.anonymize,
    )
    return 0
