"""Stubs for common interface for any image format--volume or surface, binary or xml."""

from typing import Any, Literal

from numpy import dtype, floating, float64, ndarray
from numpy.typing import DTypeLike


class FileBasedHeader:
    def get_zooms(self) -> tuple[float, ...]: ...
    
class FileBasedImage:
    affine: ndarray
    header: FileBasedHeader

    def get_fdata(
        self,
        caching: Literal['fill', 'unchanged'] = 'fill',
        dtype: DTypeLike = float64,
    ) -> ndarray[Any, dtype[floating]]: ...
