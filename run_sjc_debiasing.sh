# Sampling with score debiasing
python /path/to/sjc/run_sjc.py
--sd.prompt "a colorful toucan with a large beak"
--n_steps 10000
--lr 0.05
--sd.scale 100.0
--score_thres 8.0
--score_dynamic True
--emptiness_weight 10000
--emptiness_step 0.5
--emptiness_multiplier 20.0
--depth_weight 0
