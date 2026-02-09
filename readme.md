# Code for ``SRE-Sparsify: 3-WL-Expressive GNNs at Near-Linear Cost via Hierarchical Cut Decomposition''

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
for poly_dim in 8
do
for num_layers in 4
do
for dname in FRANKENSTEIN NCI1 NCI109 ENZYMES  
do

# RSE-Sparsify 
python run_tud.py --cfg configs/bsr_ppgn/tud.bsr_ppgn.poly.yaml --poly_dim $poly_dim  --dataname $dname 

# RSE-Dist-Sparsify 
python run_tud.py --cfg configs/bsrd_ppgn/tud.bsrd4_ppgn.poly.yaml --poly_dim $poly_dim  --dataname $dname 

done
done
done
```


### For ZINC

```bash
source ~/miniforge3/bin/activate gnn270

for poly_dim in 8
do

# RSE-Sparsify 
python run_zinc.py --cfg configs/bsr_ppgn/zinc.bsr_ppgn.poly.yaml --poly_dim $poly_dim 

# RSE-Dist-Sparsify
python run_zinc.py --cfg configs/bsrd_ppgn/zinc.bsrd4_ppgn.poly.yaml --poly_dim $poly_dim

done
```

### For ZINC-Full

```bash
source ~/miniforge3/bin/activate gnn270

for poly_dim in 8
do

python run_zincfull.py --cfg configs/bsr_ppgn/zincfull.bsr_ppgn.poly.yaml --poly_dim $poly_dim 
python run_zincfull.py --cfg configs/bsrd_ppgn/zincfull.bsrd_ppgn.poly.yaml --poly_dim $poly_dim 

done
```


### For QM9

```bash
source ~/miniforge3/bin/activate gnn270

for poly_dim in 8
do

python run_qm9_nogeo.py --cfg configs/bsr_ppgn/nogeo_qm9.bsr_ppgn.poly.yaml --poly_dim $poly_dim 
python run_qm9_nogeo.py --cfg configs/bsrd_ppgn/nogeo_qm9.bsr_ppgn.poly.yaml --poly_dim $poly_dim 

done
```

