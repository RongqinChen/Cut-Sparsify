# Code for Connectivity-Guided Sparsification of 2-FWL GNNs: Preserving Full Expressivity with Improved Efficiency

## Create environment

```bash
mamba create --name gnn270 python=3.12 pip sage -c conda-forge  -y
mamba activate gnn270
pip install torch==2.7.0  --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric==2.6.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install yacs ogb networkx torchmetrics
pip install rdkit pytorch_lightning
```


## Scripts for running experiments



### For TUD

```bash
for poly_dim in 8 12 14 16
do
for num_layers in 4
do
for dname in MUTAG PROTEINS_full ENZYMES  
do

python run_tud.py --cfg configs/sppgn/tud.sppgn.poly.yaml \
    --poly_method rrwp --poly_dim $poly_dim  --dataname $dname

done
done
done
```


### For ZINC

```bash
for poly_dim in 4 8 10 12 14
do

python run_zinc.py --cfg configs/bsr_ppgn/zinc.bsr_ppgn.poly.yaml --poly_dim $poly_dim 

done
```

### For ZINC-Full

```bash
for poly_dim in 6 8 10 12 14
do

python run_zinc.py --cfg configs/sppgn/zincfull.sppgn.poly.yaml --poly_dim $poly_dim 

done
```


### For QM9

```bash
for poly_dim in 6 8 10 12 14
do

python run_zinc.py --cfg configs/sppgn/nogeo_qm9.sppgn.poly.yaml --poly_dim $poly_dim 

done
```
