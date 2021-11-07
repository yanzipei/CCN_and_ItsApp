#C_list=('100000.0' '10000.0' '1000.0' '100.0' '10.0' '1.0' '0.1' '0.01' '0.001' '0.0001')
#C_list=('500.0' '5000.0' '50000.0')
#C_list=('600.0' '700.0' '800.0' '900.0' '1100.0' '2000.0')
#C_list=('800.0' '2000.0' '8000.0' '20000.0')
#C_list=('6000.0' '7000.0' '9000.0')
#C_list=('7200.0' '7500.0' '7800.0' '8200.0' '8500.0' '8800.0')
#C_list=('7900.0' '8100.0')
#C_list=('7920.0' '7950.0' '7980.0' '8020.0' '8050.0' '8080.0')
C_list=('7990.0' '8010.0')
for C in "${C_list[@]}"; do
  echo "C: ${C}"
  python tune_svm.py --data ./data/data_except_bgw_400.npy --log_dir ./runs/tune/svm --C "${C}"
done