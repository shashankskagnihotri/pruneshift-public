import csv
import os


#dir = "/misc/student/agnihotr/master_thesis/"
list1 = ["/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_1.1_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_1.3_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_1.5_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_2_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_4_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_8_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_16_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/orig_0_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_1.1_kd_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_1.3_kd_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_1.5_kd_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_2_kd_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_4_kd_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_8_kd_augmix/metrics.csv",
	"/misc/lmbraid19/agnihotr/thesis_pruneshift/attempt_2/l1_16_kd_augmix/metrics.csv"]

dire="/misc/lmbraid19/agnihotr/thesis_pruneshift/graphs"	
list2 = ["orig_0_augmix",
	"l1_1.1_augmix",
	"l1_1.3_augmix",
	"l1_1.5_augmix",
	"l1_2_augmix",
	"l1_4.1_augmix",
	"l1_8_augmix",
	"l1_16_augmix",
	"orig_0_augmix",
	"l1_1.1_kd_augmix",
	"l1_1.3_kd_augmix",
	"l1_1.5_kd_augmix",
	"l1_2_kd_augmix",
	"l1_4_kd_augmix",
	"l1_8_kd_augmix",
	"l1_16_kd_augmix"]

for i in range(len(list1)):
    f1 = open(list1[i], 'r')
    path = os.path.join(dire, list2[i])
    if not os.path.isdir(path):
    	os.makedirs(path)
    print(path)
    filename = path + "/metric.csv"
    print(filename)
    with open(filename, 'w') as f2:
    	#writer = csv.writer(f2)
    	for line in f1:
    	    #writer.writerows(line)
    	    f2.write(line)
