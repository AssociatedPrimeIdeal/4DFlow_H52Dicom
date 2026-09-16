"""Public package interface for 4D flow H5 to DICOM conversion."""

from .converter import array2dicom, convert_array_to_dicom, main

__all__ = ["array2dicom", "convert_array_to_dicom", "main"]
__version__ = "0.1.0"
