# Code for ``SRE-Sparsify: 3-WL-Expressive GNNs at Near-Linear Cost via Hierarchical Cut Decomposition''

## Create environment

```bash
mamba create --name gnn270 python=3.12 pip sage -c conda-forge  -y
mamba activate gnn270
pip install torch==2.7.0  --index-url https://download.pytorch.org/whl/cu128
pip install torch_geometric==2.6.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install yacs ogb networkx torchmetrics
pip install rdkit lightning
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
python run_tud.py --cfg configs/rse_ppgn/tud.rse_ppgn.poly.yaml --poly_dim $poly_dim  --dataname $dname 

# RSE-Dist-Sparsify 
python run_tud.py --cfg configs/rsed_ppgn/tud.rsed4_ppgn.poly.yaml --poly_dim $poly_dim  --dataname $dname 

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
python run_zinc.py --cfg configs/rse_ppgn/zinc.rse_ppgn.poly.yaml --poly_dim $poly_dim 

# RSE-Dist-Sparsify
python run_zinc.py --cfg configs/rsed_ppgn/zinc.rsed4_ppgn.poly.yaml --poly_dim $poly_dim

done
```

### For ZINC-Full

```bash
source ~/miniforge3/bin/activate gnn270

for poly_dim in 8
do

python run_zincfull.py --cfg configs/rse_ppgn/zincfull.rse_ppgn.poly.yaml --poly_dim $poly_dim 
python run_zincfull.py --cfg configs/rsed_ppgn/zincfull.rsed_ppgn.poly.yaml --poly_dim $poly_dim 

done
```


### For QM9

```bash
source ~/miniforge3/bin/activate gnn270

for poly_dim in 8
do

python run_qm9_nogeo.py --cfg configs/rse_ppgn/nogeo_qm9.rse_ppgn.poly.yaml --poly_dim $poly_dim 
python run_qm9_nogeo.py --cfg configs/rsed_ppgn/nogeo_qm9.rse_ppgn.poly.yaml --poly_dim $poly_dim 

done
```

### For MolHiv

```bash
source ~/miniforge3/bin/activate gnn270

for poly_dim in 8
do

python run_ogbg.py --cfg configs/sppgn/molhiv.sppgn.poly.yaml --poly_dim $poly_dim 
python run_ogbg.py --cfg configs/rse_ppgn/molhiv.rse_ppgn_poly.yaml --poly_dim $poly_dim 
python run_ogbg.py --cfg configs/rsed_ppgn/molhiv.rsed2_ppgn.poly.yaml --poly_dim $poly_dim 
python run_ogbg.py --cfg configs/rsed_ppgn/molhiv.rsed4_ppgn.poly.yaml --poly_dim $poly_dim 

done
```

### For Molecular Regresion

```bash
python run_ogbg.py --cfg configs/sppgn/molesol_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/molesol_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/sppgn/molesol_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/molesol_poly_dim_${poly_dim}_num_layers_5.log

python run_ogbg.py --cfg configs/sppgn/molfreesolv_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/molfreesolv_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/sppgn/molfreesolv_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/molfreesolv_poly_dim_${poly_dim}_num_layers_5.log

python run_ogbg.py --cfg configs/sppgn/mollipo_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/mollipo_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/sppgn/mollipo_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/mollipo_poly_dim_${poly_dim}_num_layers_5.log

```


```bash
python run_ogbg.py --cfg configs/rse_ppgn/molesol_reg.rse_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/molesol_rse_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/rse_ppgn/molesol_reg.rse_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/molesol_rse_poly_dim_${poly_dim}_num_layers_5.log

python run_ogbg.py --cfg configs/rse_ppgn/molfreesolv_reg.rse_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/molfreesolv_rse_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/rse_ppgn/molfreesolv_reg.rse_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/molfreesolv_rse_poly_dim_${poly_dim}_num_layers_5.log

python run_ogbg.py --cfg configs/rse_ppgn/mollipo_reg.rsed_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/mollipo_rse_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/rse_ppgn/mollipo_reg.rsed_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/mollipo_rse_poly_dim_${poly_dim}_num_layers_5.log

```


```bash
python run_ogbg.py --cfg configs/rse_ppgn/molesol_reg.rsed2_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/molesol_rsed2_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/rse_ppgn/molesol_reg.rsed2_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/molesol_rsed2_poly_dim_${poly_dim}_num_layers_5.log
python run_ogbg.py --cfg configs/rse_ppgn/molesol_reg.rsed4_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/molesol_rsed4_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/rse_ppgn/molesol_reg.rsed4_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/molesol_rsed4_poly_dim_${poly_dim}_num_layers_5.log

python run_ogbg.py --cfg configs/rse_ppgn/molfreesolv_reg.rsed2_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/molfreesolv_rsed2_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/rse_ppgn/molfreesolv_reg.rsed2_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/molfreesolv_rsed2_poly_dim_${poly_dim}_num_layers_5.log
python run_ogbg.py --cfg configs/rse_ppgn/molfreesolv_reg.rsed4_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/molfreesolv_rsed4_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/rse_ppgn/molfreesolv_reg.rsed4_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/molfreesolv_rsed4_poly_dim_${poly_dim}_num_layers_5.log

python run_ogbg.py --cfg configs/rse_ppgn/mollipo_reg.rsed2_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/mollipo_rsed2_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/rse_ppgn/mollipo_reg.rsed2_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/mollipo_rsed2_poly_dim_${poly_dim}_num_layers_5.log
python run_ogbg.py --cfg configs/rse_ppgn/mollipo_reg.rsed4_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 | tee -a results/mollipo_rsed4_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/rse_ppgn/mollipo_reg.rsed4_ppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 | tee -a results/mollipo_rsed4_poly_dim_${poly_dim}_num_layers_5.log

```
