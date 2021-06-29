import numpy as np

ma = np.load('metric_plus_abs_d_cv_gb_zone_5_zero_list.npy')
oa = np.load('order_arr_metric_plus_abs_d_cv_gb_zone_5_zero_list.npy')

print()

for i in oa:
    print(ma[i])
