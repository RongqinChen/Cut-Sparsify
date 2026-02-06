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
for poly_dim in 4 6 8 12 14 16
do
for num_layers in 4
do
for dname in FRANKENSTEIN NCI1 NCI109 ENZYMES  
do

CUDA_VISIBLE_DEVICES=1 python run_tud.py --cfg configs/bsr_ppgn/tud.bsr_ppgn.poly.yaml \
    --poly_method rrwp --poly_dim $poly_dim  --dataname $dname | tee results/log_${dname}_${poly_dim}_${num_layers}.txt

done
done
done
```


### For ZINC

```bash
source ~/miniforge3/bin/activate gnn270

for poly_dim in 4 6 8 10 12 14 16
do

CUDA_VISIBLE_DEVICES=0 python run_zinc.py \
    --cfg configs/bsr_ppgn/zinc.bsr_ppgn.poly.yaml \
    --poly_dim $poly_dim | tee results/log_zinc_${poly_dim}.txt

done

```

### For ZINC-Full

```bash
source ~/miniforge3/bin/activate gnn270

for poly_dim in 4 6 8 10 12 14 16
do

CUDA_VISIBLE_DEVICES=0 python run_zincfull.py \
    --cfg configs/bsr_ppgn/zincfull.bsr_ppgn.poly.yaml \
    --poly_dim $poly_dim | tee results/log_zinc_${poly_dim}.txt

done

```


### For QM9

```bash
source ~/miniforge3/bin/activate gnn270

for poly_dim in 14 12 10 8 6 4
do

CUDA_VISIBLE_DEVICES=0 python run_qm9_nogeo.py \
    --cfg configs/bsr_ppgn/nogeo_qm9.bsr_ppgn.poly.yaml \
    --poly_dim $poly_dim | tee results/log_qm9_nogeo_${poly_dim}.txt

done

```

### For PCQM4M

```bash
source ~/miniforge3/bin/activate gnn270

for poly_dim in 8
do

CUDA_VISIBLE_DEVICES=0 python run_pcqm4m.py \
    --cfg configs/bsr_ppgn/pcqm4m.bsr_ppgn.poly.yaml \
    --poly_dim $poly_dim | tee results/log_pcqm4m_${poly_dim}.txt

done

```

### For Pep-func

```bash
source ~/miniforge3/bin/activate gnn270

for poly_dim in 8
do

CUDA_VISIBLE_DEVICES=0 python run_peptides_func.py \
    --cfg configs/bsr_ppgn/pep_func.bsr_ppgn.poly.yaml \
    --poly_dim $poly_dim | tee results/log_func_${poly_dim}.txt

done

```
