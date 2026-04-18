source ~/miniforge3/bin/activate gnn270

export CUDA_VISIBLE_DEVICES=1

for poly_dim in 4
do

python run_peptides_struct.py --cfg configs/rsed_ppgn/pep_str.rsed2_ppgn.poly.yaml --poly_dim $poly_dim 

done
