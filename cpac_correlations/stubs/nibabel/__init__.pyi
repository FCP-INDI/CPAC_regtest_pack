"""Type stubs for the parts of Nibabel that we use."""

from nibabel.filebasedimages import FileBasedImage
from nibabel.filename_parser import FileSpec


def load(filename: FileSpec, **kwargs) -> FileBasedImage: ...

__all__ = ["load"]
