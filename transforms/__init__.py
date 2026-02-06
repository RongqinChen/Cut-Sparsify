# Transforms for feature augmentation of graph neural network inputs
from .compute_conn_and_poly import ConnAndPoly
from .compute_distances import RD, SPD
from .compute_polynomial import Polynomials

# Dataset-specific transforms
from .qm9_input_transform import QM9InputTransform

# 2-Folklore Weisfeiler-Leman (2-FWL) transforms
from .compute_2fwl import K2FWLTransform
from .compute_2fwl_connsp import K2FWLConnSpTransform
from .compute_2fwl_conndistsp import K2FWLConnDistSpTransform
from .compute_2fwl_bsr import BSR2FWLTransform


# Transform registry: maps string identifiers to transform classes
# This enables dynamic transform selection via configuration
transform_dict = {
    # Input preprocessing transforms
    "qm9_input_transform": QM9InputTransform,

    # Polynomial-based transforms
    "poly": Polynomials,  # Graph adjacency/Laplacian polynomial features
    "conn_poly": ConnAndPoly,  # Local connectivity and polynomial features

    # 2-Folklore Weisfeiler-Leman transforms
    "2fwl": K2FWLTransform,  # Standard 2-FWL: processes all 3-tuples (a,b,c)
    "2fwl_connsp": K2FWLConnSpTransform,  # Connectivity-guided sparsification
    "2fwl_conndistsp": K2FWLConnDistSpTransform,  # Connectivity+distance co-guided sparsification
    "2fwl_bsr": BSR2FWLTransform,  # Block-SPQR-based sparsification

    # Distance-based transforms
    "RD": RD,  # Resistance Distance
    "SPD": SPD,  # Shortest Path Distance
}
