source ~/miniforge3/bin/activate gnn270

for poly_dim in 8 12
do
for num_layers in 4
do
for dname in FRANKENSTEIN NCI1 NCI109 ENZYMES  
do

python run_tud.py --cfg configs/cat_mlp/tud.sppgn.poly.yaml --poly_dim $poly_dim  --dataname $dname --specified_run 0 | tee -a results/tud_sppgn_poly_dim${poly_dim}_numlayers${num_layers}_${dname}.log

done
done
done

for poly_dim in 8 12
do

python run_zinc.py --cfg configs/cat_mlp/tud.sppgn.poly.yaml --poly_dim $poly_dim --specified_run 0 | tee -a results/zinc_sppgn_poly_dim${poly_dim}.log 

done
