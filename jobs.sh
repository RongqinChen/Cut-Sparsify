source ~/miniforge3/bin/activate gnn270

for poly_dim in 4
do
for num_layers in 4
do
for dname in FRANKENSTEIN NCI1 NCI109 ENZYMES  
do

# CO-Sparsify
python run_tud.py --cfg configs/sppgn/tud.sppgn.poly.yaml --poly_dim $poly_dim  --dataname $dname --specified_run 0 | tee -a results/tud_sppgn_poly_dim${poly_dim}_numlayers${num_layers}_${dname}.log

# RSE-Sparsify 
python run_tud.py --cfg configs/rse_ppgn/tud.rse_ppgn.poly.yaml --poly_dim $poly_dim  --dataname $dname --specified_run 0 | tee -a results/tud_rse_ppgn_poly_dim${poly_dim}_numlayers${num_layers}_${dname}.log

# RSE-Dist-Sparsify 
python run_tud.py --cfg configs/rsed_ppgn/tud.rsed4_ppgn.poly.yaml --poly_dim $poly_dim  --dataname $dname --specified_run 0 | tee -a results/tud_rsed_ppgn_poly_dim${poly_dim}_numlayers${num_layers}_${dname}.log

done
done
done

for poly_dim in 4
do

# CO-Sparsify 
python run_zinc.py --cfg configs/sppgn/zinc.sppgn.poly.yaml --poly_dim $poly_dim --specified_run 0 | tee -a results/zinc_sppgn_poly_dim${poly_dim}.log 

# RSE-Sparsify 
python run_zinc.py --cfg configs/rse_ppgn/zinc.rse_ppgn.poly.yaml --poly_dim $poly_dim --specified_run 0 | tee -a results/zinc_rse_ppgn_poly_dim${poly_dim}.log 

# RSE-Dist-Sparsify
python run_zinc.py --cfg configs/rsed_ppgn/zinc.rsed4_ppgn.poly.yaml --poly_dim $poly_dim --specified_run 0 | tee -a results/zinc_rsed_ppgn_poly_dim${poly_dim}.log

done

for poly_dim in 8
do

# CO-Sparsify 
python run_zincfull.py --cfg configs/sppgn/zincfull.sppgn.poly.yaml --poly_dim $poly_dim --specified_run 0 | tee -a results/zincfull_sppgn_poly_dim${poly_dim}.log

# RSE-Sparsify 
python run_zincfull.py --cfg configs/rse_ppgn/zincfull.rse_ppgn.poly.yaml --poly_dim $poly_dim --specified_run 0 | tee -a results/zincfull_rse_ppgn_poly_dim${poly_dim}.log

# RSE-Dist-Sparsify
python run_zincfull.py --cfg configs/rsed_ppgn/zincfull.rsed_ppgn.poly.yaml --poly_dim $poly_dim --specified_run 0 | tee -a results/zincfull_rsed_ppgn_poly_dim${poly_dim}.log

done

for poly_dim in 8
do

# CO-Sparsify 
python run_qm9_nogeo.py --cfg configs/sppgn/nogeo_qm9.sppgn.poly.yaml --poly_dim $poly_dim --specified_run 0 | tee -a results/qm9_nogeo_sppgn_poly_dim${poly_dim}.log

# RSE-Sparsify 
python run_qm9_nogeo.py --cfg configs/rse_ppgn/nogeo_qm9.rse_ppgn.poly.yaml --poly_dim $poly_dim --specified_run 0 | tee -a results/qm9_nogeo_rse_ppgn_poly_dim${poly_dim}.log

# RSE-Dist-Sparsify
python run_qm9_nogeo.py --cfg configs/rsed_ppgn/nogeo_qm9.rse_ppgn.poly.yaml --poly_dim $poly_dim --specified_run 0 | tee -a results/qm9_nogeo_rsed_ppgn_poly_dim${poly_dim}.log

done

for poly_dim in 8
do

python run_ogbg.py --cfg configs/sppgn/molesol_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 --specified_run 0 | tee -a results/molesol_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/sppgn/molesol_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 --specified_run 0 | tee -a results/molesol_poly_dim_${poly_dim}_num_layers_5.log

python run_ogbg.py --cfg configs/sppgn/molfreesolv_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 --specified_run 0 | tee -a results/molfreesolv_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/sppgn/molfreesolv_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 --specified_run 0 | tee -a results/molfreesolv_poly_dim_${poly_dim}_num_layers_5.log

python run_ogbg.py --cfg configs/sppgn/mollipo_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 3 --specified_run 0 | tee -a results/mollipo_poly_dim_${poly_dim}_num_layers_3.log
python run_ogbg.py --cfg configs/sppgn/mollipo_reg.sppgn.poly.yaml --poly_dim $poly_dim --num_layers 5 --specified_run 0 | tee -a results/mollipo_poly_dim_${poly_dim}_num_layers_5.log

done
