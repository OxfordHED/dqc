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
from dqc.xc import function_xc

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

        # Map the different components of the xc to the right functions
        components = [comp.strip() for comp in xc_names.split("+")]

        formatted_components = []

        def is_number(str_inp):
            try:
                float(str_inp)
                return True
            except ValueError:
                return False

        def map_libxc_or_fnxc(str_inp):
            if str_inp.startswith("fn_"):
                return str_inp
            return f"get_libxc('{str_inp}')"

        for comp in components:
            if "*" in comp:
                comp_parts = comp.split("*")
                comp_parts = [
                    cp if is_number(cp.strip())
                    else map_libxc_or_fnxc(cp.strip())
                    for cp in comp_parts
                ]
                formatted_components.append("*".join(comp_parts))
            else:
                formatted_components.append(comp)

        new_xcstr = " + ".join(formatted_components)

        # evaluate the expression and return the xc
        glob = {"get_libxc": get_libxc, "fn_linear": function_xc.get_linear}
        return eval(new_xcstr, glob)
    else:
        xc_sum = get_libxc(xc_names[0])
        for xc_name in xc_names[1:]:
            xc_sum += get_libxc(xc_name)
        return xc_sum
