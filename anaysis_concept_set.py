import math
import matplotlib.pyplot as plt
import numpy as np
import os

# random_class = ['n02917067', 'n02802426', 'n02483362', 'n02487347', 'n02494079', 'n02486410', 'n13054560',
#                 'n12998815', 'n02835271', 'n03792782']

# random_class = ['n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021',
#                 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748',
#                 'n01753488', 'n01755581', 'n01756291', 'n02276258', 'n02277742', 'n02279972', 'n02280649',
#                 'n02281406', 'n02281787', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075',
#                 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02085620', 'n02085782',
#                 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394',
#                 'n02088094','n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041',
#                 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849',
#         'n02013706','n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062',
#           'n01776313']

random_class = ['n01728572', 'n02276258', 'n02123045', 'n02128385', 'n02085620', 'n01443537', 'n02002724', 'n01773157',
                'n02483362']

snake = ['n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021',
         'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748',
         'n01753488', 'n01755581', 'n01756291']

butterfly = ['n02276258', 'n02277742', 'n02279972', 'n02280649',
             'n02281406', 'n02281787']

cat = ['n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075',
       'n02125311', 'n02127052']

leopard = ['n02128385', 'n02128757', 'n02128925']

dog = ['n02085620', 'n02085782',
       'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394',
       'n02088094']

fish = ['n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041']

bird = ['n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849',
        'n02013706']

spider = ['n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062',
          'n01776313']

monkey = ['n02483362', 'n02487347', 'n02494079', 'n02486410']

lizard = ['n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333',
          'n01693334', 'n01694178', 'n01695060']

wall_g = ['n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777']

fox = ['n02119022', 'n02119789', 'n02120079', 'n02120505']

li = ['n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366']

ox = ['n02403003', 'n02408429', 'n02410509']

sheep = ['n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022']

mushroom = ['n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560']

deleted_node_time_one_sets = []

node_dir = 'best_node_pools_npy_dir_t_830'

# for random_class_item in random_class:
#     deleted_node_time_one_sets.append(
#         np.load(node_dir+'/deleted_node_js_marker_toc_set_%s.npy' % random_class_item))

all_node_set = set(list(range(9216)))

# concepts = [snake, butterfly, cat, leopard, dog, fish, bird, spider, monkey]
# concept_names = ['snake', 'butterfly', 'cat', 'leopard', 'dog', 'fish', 'bird', 'spider', 'monkey']
#
concepts = [snake, butterfly, cat, leopard, dog, fish, bird, spider]
concept_names = ['sna', 'but', 'cat', 'le', 'dog', 'fish', 'bird', 'spi']
# concepts = [lizard, wall_g, fox, li, ox, sheep,
#             mushroom]
# concept_names =['liza', 'wall',
#                  'fox', 'li', 'ox', 'she', 'mush']

inter_num_mean_list = []

# for i in range(len(concepts)):
#     concepts_inter_set = all_node_set - set(np.load(
#         node_dir+'/deleted_node_wad_marker_toc_sc_set_%s.npy' % concepts[i][0]))
#     for m in range(1, len(concepts[i])):
#         concepts_inter_set = concepts_inter_set.intersection(all_node_set - set(np.load(
#             node_dir+'/deleted_node_wad_marker_toc_sc_set_%s.npy' % concepts[i][m])))
#
#     print('%s: %d' % (concept_names[i], len(concepts_inter_set)))

# op = 'cal_inter_set'
op = 'plot'
# op = 'none'

if op == 'cal_inter_set':
    for i in range(len(concepts)):
        for j in range(len(concepts)):

            inter_num_list = []

            for m in range(len(concepts[i])):
                for n in range(len(concepts[j])):
                    if m == n and i == j:
                        continue
                    inter_num = len(all_node_set - (set(np.load(
                        node_dir + '/deleted_node_wad_marker_toc_sc_set_%s.npy' % concepts[i][
                            m]))).intersection(
                        (all_node_set - set(
                            np.load(
                                node_dir + '/deleted_node_wad_marker_toc_sc_set_%s.npy' % concepts[j][n])))))
                    inter_num_list.append(inter_num)

            inter_num_mean = np.mean(np.array(inter_num_list))
            inter_num_mean_list.append(inter_num_mean)
            # print(len(inter_num_list))
            print('%s-%s: %.2f' % (concept_names[i], concept_names[j], inter_num_mean))
            with open(r'anaysis_concept_set_0829.txt', 'a+') as f:
                f.write('%s %s %.2f\n' % (concept_names[i], concept_names[j], inter_num_mean))

if op == 'plot':
    inter_num_dic = {}
    with open('anaysis_concept_set_0829.txt', 'r') as f1:
        infos = f1.readlines()
        for i in range(0, len(infos)):
            info = infos[i].split(' ')
            class_1 = info[0]
            if not inter_num_dic.__contains__(class_1):
                inter_num_dic[class_1] = {}
            inter_num_dic[class_1][info[1]] = float(info[2].replace('\n', ''))

    # print(inter_num_dic)

    inter_plot_height = int(math.sqrt(len(concepts)))
    inter_plot_width = int(len(concepts) / inter_plot_height)
    if inter_plot_height * inter_plot_width < len(concepts):
        inter_plot_width += 1

    inter_plot_height = 1
    inter_plot_width = 1

    axs = []
    fig = plt.figure()
    count = 0
    for i in range(len(concepts)):
        if i != 2:
            continue

        axs.append(fig.add_subplot(inter_plot_height, inter_plot_width, count + 1))
        count += 1

    x = concept_names
    y = []
    for key in x:
        sub_y = []
        for sub_key in x:
            sub_y.append(inter_num_dic[key][sub_key])
        y.append(sub_y)

    count = 0
    for i in range(len(concepts)):
        if x[i] != 'cat':
            continue
        # i = count
        axs[count].plot(x, y[i], color='black',)
        # axs[count].set_title(x[i],fontsize=18)
        axs[count].set_ylim([7000, 8200])
        axs[count].set_ylabel('结构相似比例',fontsize=21)
        axs[count].set_xlabel('物种简称',fontsize=21)

        for j in range(len(concepts)):
            axs[count].plot([j, j], [0, y[i][j]], color='black', linestyle='--')
        count += 1

    plt.yticks(size=18)
    plt.xticks(size=18)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.legend()
    plt.show()
    plt.savefig('D:\\analysis_concept.svg', format='svg')
