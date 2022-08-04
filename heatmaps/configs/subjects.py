import os

from itertools import chain
from string import ascii_lowercase


def cpac_sub(sub):
    """
    Function to convert a string from f"sub-{sub_number}{ses_letter}" to
    f"sub-{sub_number}_ses-{ses_number}"

    Parameter
    ---------
    fmriprep_sub: str

    Returns
    -------
    sub: str

    Example
    -------
    >>> print(cpac_sub("sub-0025427a"))
    sub-0025427_ses-1
    """
    return(f"{sub[:-1]}_ses-{str(ascii_lowercase.find(sub[-1])+1)}")


def fmriprep_sub(sub):
    """
    Function to convert a string from f"sub-{sub_number}_ses-{ses_number}" to
    f"sub-{sub_number}{ses_letter}"

    Parameter
    ---------
    sub: str

    Returns
    -------
    fmriprep_sub: str

    Example
    -------
    >>> print(fmriprep_sub("sub-0025427_ses-1"))
    sub-0025427a
    """
    return(f"{sub.split('_')[0]}{ascii_lowercase[int(sub[-1])-1]}")


def generate_subject_list_for_directory(path, old_outputs_software="C-PAC"):
    """
    Function to take a path and return a subject list.

    Parameter
    ---------
    path: str

    old_outputs_software: str, optional, default="C-PAC"

    Returns
    -------
    sub_list: list
    """
    output = os.path.join(path, "output")
    sub_ses_list = list(chain.from_iterable([[
        d for d in os.listdir(
            os.path.join(output, o)
        ) if all([
            os.path.isdir(os.path.join(output, o, d)),
            d not in ["log", "logs"]
        ])
    ] for o in os.listdir(output)]))
    return(sessions_together([
        cpac_sub(s) if s[
            -1
        ] in ascii_lowercase else s for s in sub_ses_list
    ]))


def generate_subject_list_for_range(
    subject_start_stop,
    session_start_stop=None
):
    """
    Function to create a subject list for a given range. All values are
    inclusive.

    Parameters
    ----------
    subject_start_stop: 2-tuple of integers (start, stop) or list of specific
    values

    session_start_stop: 2-tuple of integers (start, stop) or list of specific
    values or None

    Returns
    -------
    List of strings

    Example
    -------
    >>> generate_subject_list_for_range((25427,25428), (1,2))
    ['sub-0025427_ses-1', 'sub-0025428_ses-1', 'sub-0025427_ses-2', 'sub-0025428_ses-2']
    """
    return([
        f'sub-00{sub}{ses_string}' for ses_string in ([
            f'_ses-{ses}' for ses in _expand_range(
                session_start_stop
            )
        ] if session_start_stop else [
            ''
        ]) for sub in _expand_range(subject_start_stop)
    ])


def sessions_together(sub_list):
    """
    Function to sort by session then by subject

    Parameter
    ---------
    sub_list: list of str

    Returns
    -------
    sub_list: list of str

    Example
    -------
    >>> sub_list = [
    ...    'sub-0025427_ses-1', 'sub-0025427_ses-2', 'sub-0025428_ses-1'
    ... ]
    >>> print(sessions_together(sub_list))
    ['sub-0025427_ses-1', 'sub-0025428_ses-1', 'sub-0025427_ses-2']
    """
    sub_list.sort()
    sub_list.sort(key=lambda x: x.split("ses-")[-1])
    return(sub_list)


def _expand_range(tuple_or_list):
    """
    Function to expand an inclusive tuple to a range or return a literal list

    Parameter
    ---------
    tuple_or_list: 2-tuple of integers or list

    Returns
    -------
    list
    """
    return(
        list(
            range(
                tuple_or_list[0],
                tuple_or_list[1] + 1) if all([
                isinstance(tuple_or_list, tuple),
                len(tuple_or_list)==2,
                *[isinstance(v, int) for v in tuple_or_list]
            ]) else tuple_or_list
        )
    )
