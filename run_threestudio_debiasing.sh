# Sampling with score debiasing
cd threestudio
python launch.py --config configs/dreamfusion-if.yaml --train --gpu 0 system.prompt_processor.prompt="a colorful toucan with a large beak" system.guidance.grad_clip=[0,0.5,2.0,10000]