bsub -W 10:00 -n 1 -M 64000 -gpu "num=1" -q gpu-v100 -R "span[hosts=1]" -Is bash

proxy_on # turn on to connect the internet for jupyter notebook

module load anaconda3

module load cuda/11.7

source activate tensorflow-2 # this env only for prott5 embedding, if you want to run fine-tuning pt5, pls let me know

jupyter notebook --ip '0.0.0.0' # click on the link http://bmi-{cluster-id}...


