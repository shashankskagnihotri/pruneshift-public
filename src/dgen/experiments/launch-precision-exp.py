import os
trainer = "/home/saikiat/myrepo/domain-gen/dgen/trainer.py"

cmd = "tsub --gpus 2 --cpus 8 {} -- --data-path /misc/scratchSSD2/datasets/ILSVRC2012-100 --num-classes 100 --gpus 2 -b 128  --id={}  -j 8 --distributed_backend ddp"

exp_name = "72b1-amix-in-100-ddp-fp32"
torun = cmd.format(trainer, exp_name, 0) + " --precision 32"
print(torun)
os.system(torun)

exp_name = "72b2-amix-in-100-ddp-fp16"
torun = cmd.format(trainer, exp_name, 1) + " --precision 16"
print(torun)
os.system(torun)

#for i, num in enumerate(tv_layers):
#    torun = cmd.format(trainer, num, exp_name, i)
#    print(torun)
#    os.system(torun)
