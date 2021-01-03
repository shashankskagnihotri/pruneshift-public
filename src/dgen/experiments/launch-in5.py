import os
trainer = "/home/saikiat/myrepo/domain-gen/dgen/trainer.py"

cmd = "tsub {} -- --data-path /misc/scratchSSD2/datasets/ILSVRC2012-5 --num-classes 5 --gpus 1 -b 64 --num-tv-layers {} --tv {} --id={} --ver={}"

tv_layers = [1] #2, 3, 5]

TV_FACTOR = 0.02
exp_name = "71a2-amix-vary-num-tv-layers-1-tv-factor-0.02-epochs-180"
for i, num in enumerate(tv_layers):
    torun = cmd.format(trainer, num, TV_FACTOR, exp_name, i) + " --max-epochs 180"
    print(torun)
    os.system(torun)
