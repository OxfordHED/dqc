from __future__ import annotations

import re
import warnings
from typing import List
try:
    import pylibxc
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn("Failed to import pylibxc. Might not be able to use xc.")
from dqc.xc.base_xc import BaseXC, ZeroXC
from dqc.xc.libxc import get_libxc

__all__ = ["get_xc"]

def get_xc(xc_names: str | List[str]) -> BaseXC:
    """
    Returns the XC object based on the expression in xc_names.

    Arguments
    ---------
    xc_names: str | list[str]
        The expression of the xc string, e.g. ``"lda_x + gga_c_pbe"`` or ``["lda_x", "lda_c_pw"]``
        where the variable name will be replaced by the LibXC object

    Returns
    -------
    BaseXC
        XC object based on the given expression
    """
    if not xc_names:
        return ZeroXC()
    elif isinstance(xc_names, str):
        # wrap the name of xc with "get_libxc"
        pattern = r"([a-zA-Z_$][a-zA-Z_$0-9]*)"
        new_xcstr = re.sub(pattern, r'get_libxc("\1")', xc_names)

        # evaluate the expression and return the xc
        glob = {"get_libxc": get_libxc}
        return eval(new_xcstr, glob)
    else:
        return sum(get_libxc(xc_name) for xc_name in xc_names)