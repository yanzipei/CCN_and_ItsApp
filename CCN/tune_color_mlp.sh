lambda=('1e-2' '1e-3' '1e-4' '1e-5')
for l0 in "${lambda[@]}"; do
  for l1 in "${lambda[@]}"; do
    for l2 in "${lambda[@]}"; do
      echo "lambda: [${l0}, ${l1}, ${l2}]"
      python tune_color_mlp.py --data ./data/original_data.npy --log_dir ./runs/tune/color_mlp/original_data --l "${l0}" "${l1}" "${l2}" --hidden_dim 18 --epochs 5000 --lr 0.1 --wd 0
    done
  done
done
