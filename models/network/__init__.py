from .ppgn import PPGN
from .sppgn import SPPGN
from .cat_mlp_2fwl import CatMLP_2FWLGNN

network_dict = {
    "ppgn": PPGN,
    "sppgn": SPPGN,
    "mlp_prod_2fwl": SPPGN,
    "cat_mlp_2fwl": CatMLP_2FWLGNN,
}
