source .venv/bin/activate;

python3 main.py --config-name exp1_PPO_wrapper;
python3 main.py --config-name exp2_PPO_no_wrapper;
python3 main.py --config-name exp3_SAC_wrapper;
python3 main.py --config-name exp4_SAC_no_wrapper;
python3 main.py --config-name exp5_PPO_wrapper_thickness2;
python3 main.py --config-name exp6_PPO_wrapper_maxcolor50;
python3 main.py --config-name exp7_PPO_wrapper_drawfor50;
python3 main.py --config-name exp8_PPO_wrapper_drawfor300;
python3 main.py --config-name exp8_PPO_wrapper_drawfor300;