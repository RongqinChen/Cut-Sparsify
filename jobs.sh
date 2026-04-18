source ~/miniforge3/bin/activate gnn270

for poly_dim in 8
do

python run_peptides_struct.py --cfg configs/rsed_ppgn/pep_str.rse_ppgn.poly.yaml --poly_dim $poly_dim 

done
