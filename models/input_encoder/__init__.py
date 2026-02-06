from .dummy_encoder import DummyEdgeEncoder, DummyNodeEncoder
from .linear_encoder import LinearEdgeEncoder, LinearNodeEncoder
from .none_encoder import NoneEncoder
from .ogb_encoder import OGBEdgeEncoder, OGBNodeEncoder
from .qm9_encoder import QM9EdgeEncoder, QM9NodeEncoder
from .type_dict_encoder import TypeDictEdgeEncoder, TypeDictNodeEncoder
from .identity_encoder import IdentityNodeEncoder, IdentityEdgeEncoder, IdentityPEEncoder


node_encoder_dict = {
    "dummy": DummyNodeEncoder,
    "linear": LinearNodeEncoder,
    "none": NoneEncoder,
    "ogb": OGBNodeEncoder,
    "qm9": QM9NodeEncoder,
    "type_dict": TypeDictNodeEncoder,
    "identity": IdentityNodeEncoder,
    None: NoneEncoder,
}


edge_encoder_dict = {
    "dummy": DummyEdgeEncoder,
    "linear": LinearEdgeEncoder,
    "none": NoneEncoder,
    "ogb": OGBEdgeEncoder,
    "qm9": QM9EdgeEncoder,
    "type_dict": TypeDictEdgeEncoder,
    "identity": IdentityEdgeEncoder,
    None: NoneEncoder,
}

pe_encoder_dict = {
    'identity': IdentityPEEncoder,
    "linear": LinearEdgeEncoder,
    None: NoneEncoder,
}


__all__ = ['node_encoder_dict', 'edge_encoder_dict', 'pe_encoder_dict']
