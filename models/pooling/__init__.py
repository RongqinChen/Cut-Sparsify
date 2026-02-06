from .adaptive_dense_pooling import AdaptiveDiagOffdiagAvgPooling, AdaptiveDiagOffdiagSumPooling
from .dense_pooling import DiagOffdiagAvgPooling, DiagOffdiagSumPooling
from .sparse_pooling import GraphAvgPooling, GraphSumPooling, GraphMaxPooling

dense_pooling_dict = {"avg": DiagOffdiagAvgPooling, "sum": DiagOffdiagSumPooling,
                      "adpavg": AdaptiveDiagOffdiagAvgPooling, "adpsum": AdaptiveDiagOffdiagSumPooling}
sparse_pooling_dict = {"avg": GraphAvgPooling, "sum": GraphSumPooling, "max": GraphMaxPooling}

__all__ = ["dense_pooling_dict", "sparse_pooling_dict"]
