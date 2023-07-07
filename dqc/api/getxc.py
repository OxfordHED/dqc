import re
import warnings
try:
    import pylibxc
except (ImportError, ModuleNotFoundError) as e:
    warnings.warn("Failed to import pylibxc. Might not be able to use xc.")
from dqc.xc.base_xc import BaseXC
from dqc.xc.libxc import get_libxc

__all__ = ["get_xc"]

def get_xc(xcstr: str) -> BaseXC:
    """
    Returns the XC object based on the expression in xcstr.

    Arguments
    ---------
    xcstr: str
        The expression of the xc string, e.g. ``"lda_x + gga_c_pbe"`` where the
        variable name will be replaced by the LibXC object

    Returns
    -------
    BaseXC
        XC object based on the given expression
    """
    # wrap the name of xc with "get_libxc"
    pattern = r"([a-zA-Z_$][a-zA-Z_$0-9]*)"
    new_xcstr = re.sub(pattern, r'get_libxc("\1")', xcstr)

    # evaluate the expression and return the xc
    glob = {"get_libxc": get_libxc}
    return eval(new_xcstr, glob)
