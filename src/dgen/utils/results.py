import pandas as pd

import gnuplotlib as gp

results = pd.read_csv("metrics.csv")
print("-- Summary")
print(results)


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


clean_acc = results['clean_test_acc'].dropna().values[0]
print("-- Clean test err: {:.3f}".format(100 - clean_acc*100))
c_accs = {}
for c in CORRUPTIONS:
    c_accs[c] = []
    for sev in range(1,6):
        key = "corr_{}_severity_{}_test_acc".format(c,sev)
        item = results[key].dropna().values
        #dropna().values()
        assert(len(item) == 1)
        acc = item[0]
        c_accs[c].append(100 - acc*100)

df = pd.DataFrame(c_accs)



print("-- Mean corr test err: {:.3f}".format(df.values.mean()))

print("-- Corr Errs")
print(df)


print("-- Corr err means")
print(df.mean())



