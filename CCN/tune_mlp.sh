#hidden_dims=('3' '6' '9' '12' '15' '18' '21' '24')
#
#for hiddem_dim in "${hidden_dims[@]}"; do
#    echo "hiddem_dim: ${hiddem_dim}"
#    python tune_mlp.py --data ./data/data.npy --log_dir ./runs/tune/mlp/mish --hidden_dim "${hiddem_dim}"
#done

##lr_list=('1e-04' '1e-05' '1e-06')
#lr_list=('1e-01' '1e-02' '1e-03')
#wd_list=('1e-05' '1e-06' '0')
#
#for lr in "${lr_list[@]}"; do
#  for wd in "${wd_list[@]}"; do
#      echo "lr: ${lr}, wd: ${wd}"
#        python tune_mlp.py --data ./data/original_data.npy --log_dir ./runs/tune/mlp/original_data --hidden_dim 18 --epochs 5000 --lr "${lr}" --wd "${wd}"
#    done
#done


hidden_dims=('3' '6' '9' '12' '15' '18' '21' '24')

for hiddem_dim in "${hidden_dims[@]}"; do
    echo "hiddem_dim: ${hiddem_dim}"
    python tune_mlp.py --data ./data/original_data.npy --log_dir ./runs/tune/mlp/original_data --hidden_dim "${hiddem_dim}" --epochs 5000 --lr 0.1 --wd 0
done