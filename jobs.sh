source ~/miniforge3/bin/activate gnn270

for poly_dim in 8
do

python run_ogbg.py --cfg configs/sppgn/molhiv.sppgn.poly.yaml --poly_dim $poly_dim 
# python run_ogbg.py --cfg configs/rsed_ppgn/nogeo_qm9.rse_ppgn.poly.yaml --poly_dim $poly_dim 

done

