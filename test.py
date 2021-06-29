import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.spatial import distance
# import torch
from scipy.stats import wasserstein_distance


# from calculate2 import cal_cos
# from calculate5 import mkdir


def cal_cos(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def mkdir(path):
    import os
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)

    if not is_exists:
        os.makedirs(path)
        # print(path + ' success')
        return True
    else:
        # print(path + ' existed')
        return False


# # a = [0, 1, 3]
# # b = [0, 2, 4]
#
# import calculate2

# a = [10e-7, 10e-7, 10e-7, 10e-7]
# b = [1, 4, 2, 3]
#
# # print(b[-4:0])
#
# arr = np.array([1, 2, 3, 4])
#
# print(arr[np.array([3, 2])])
#
# # print(np.corrcoef(arr))
#
# # c = np.ones((1,4)).tolist()[0]
# # c = list(range(len(a)))
# #
# # print(c)
# #
# # print(wasserstein_distance(c, c, a, b))
# # print(calculate2.cal_jsd_list_between_tensors())
# # print(cal_jsd2(np.array(a), np.array(b)))
# # print(cal_jsd3(np.array(a), np.array(b)))
# # print(distance.jensenshannon(a, b) * distance.jensenshannon(a, b))
#
#
# # def cal_inter_num(a, b):
# #     return len(set(a).intersection(set(b)))
# #
# #
# # b1 = np.load('best_node_pools1_wad_marker_toc_sc_set_n02128385_t846.npy').tolist()
# # b2 = np.load('best_node_pools1_wad_marker_toc_sc_set_n02128385_t847.npy').tolist()
# #
# # print(len(b1))
# # print(len(b2))
# #
# # print(cal_inter_num(b1, b2))
#
# arr = np.array([[1, 0], [0, 1]])
#
# print(cal_cos(np.array([1, 1]), arr[0]))
# print(cal_cos(np.array([1, 1]), arr[1]))
# print(cal_cos(np.array([1, 1]), arr[1] + arr[0]))

# all_classes = np.load('n02105855_ts_cp_classes.npy')
# print(all_classes)
#
# all_classes = np.load('n04487394_cp_classes.npy')
# print(all_classes)

# ['n02104029' 'n07695742' 'n04192698' 'n01872401' 'n01829413' 'n04557648'
#  'n03125729' 'n02988304' 'n02009912' 'n04487394' 'n02107312' 'n02134418'
#  'n02276258' 'n07892512' 'n02128925' 'n03041632' 'n03207743' 'n02398521'
#  'n02006656' 'n02481823' 'n02093428' 'n02917067' 'n01491361' 'n02815834'
#  'n04357314' 'n02098286' 'n01843065' 'n02113023' 'n04389033' 'n03447447'
#  'n03444034' 'n02109047' 'n02102973' 'n03445924' 'n01641577' 'n03018349'
#  'n01806567' 'n01748264' 'n03482405' 'n04515003' 'n02422106' 'n02119789'
#  'n02536864' 'n03498962' 'n02869837' 'n02114712' 'n03673027' 'n03944341'
#  'n13133613' 'n02422699' 'n02087046' 'n01978455' 'n02071294' 'n04356056'
#  'n03676483' 'n04596742' 'n01632777' 'n04350905' 'n04209133' 'n03598930'
#  'n03250847' 'n01667778' 'n02102177' 'n02317335' 'n01582220' 'n04149813'
#  'n03483316' 'n02786058' 'n02486261' 'n02027492' 'n10148035' 'n01798484'
#  'n04264628' 'n13044778' 'n01498041' 'n07715103' 'n02441942' 'n01930112'
#  'n07714990' 'n02727426' 'n03891251' 'n01768244' 'n02484975' 'n04162706'
#  'n03785016' 'n03729826' 'n01775062' 'n02100735' 'n02930766' 'n03271574'
#  'n02892767' 'n01833805' 'n02101556' 'n03000247' 'n07613480' 'n02086240'
#  'n02129604' 'n03481172' 'n02099267' 'n03887697']

# n02105855

# arr = np.array([[1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]])
# weight = np.array([[2, 2, 2], [2, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6],
#                    [7, 7, 7]])
# pools = np.array([[0, 1, 2], [1, 1, 2], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
#
# print(arr.shape)
# print(weight.shape)
# print(pools.shape)
#
# res = []
# for i in range(6):
#     res.append(weight[i] * arr[:, pools[i]])
#
# print(res)
#
# for i in range(6):
#     for j in range(3):
#         arr[j, i] = np.sum(res[i][j])
#
# print('---')
# print(arr)
# print('---')

# ['n03325584', 'n04376876', 'n03476684', 'n04265275', 'n04238763', 'n02361337', 'n01698640', 'n03697007', 'n07565083',
#  'n04285008', 'n03160309', 'n03065424', 'n03895866', 'n02105505', 'n04553703', 'n02655020', 'n03871628', 'n01532829',
#  'n02687172', 'n02410509', 'n04286575', 'n04033901', 'n03376595', 'n02104029', 'n02102480', 'n02437616', 'n02980441',
#  'n03814906', 'n02672831', 'n01984695', 'n03967562', 'n02086079', 'n02168699', 'n07730033', 'n02123394', 'n04116512',
#  'n03042490', 'n03188531', 'n04482393', 'n04591713', 'n02229544', 'n02109961', 'n01728572', 'n03109150', 'n07695742',
#  'n04209239', 'n04192698', 'n03630383', 'n01820546', 'n01877812', 'n07753275', 'n01828970', 'n07697537', 'n04258138',
#  'n01806143', 'n02328150', 'n03496892', 'n03617480', 'n03000134', 'n03529860', 'n04070727', 'n03259280', 'n02128385',
#  'n01943899', 'n02927161', 'n02403003', 'n02107574', 'n02119022', 'n01843383', 'n03908714', 'n01847000', 'n03868863',
#  'n04296562', 'n01685808', 'n04476259', 'n01872401', 'n02264363', 'n02108422', 'n04270147', 'n01829413', 'n03595614',
#  'n02444819', 'n02098105', 'n04039381', 'n02128757', 'n04251144', 'n02791124', 'n02364673', 'n03794056', 'n04023962',
#  'n03873416', 'n01773549', 'n13037406', 'n04557648', 'n02111500', 'n03763968', 'n03642806', 'n02097658', 'n03538406',
#  'n03187595', 'n01631663', 'n03825788', 'n02096177', 'n01950731', 'n03127925', 'n02091467', 'n03125729', 'n02112706',
#  'n02231487', 'n02974003', 'n07932039', 'n07720875', 'n02130308', 'n02102040', 'n02799071', 'n02105251', 'n02111277',
#  'n02892201', 'n01695060', 'n03637318', 'n02089078', 'n02860847', 'n07749582', 'n02804414', 'n02114548', 'n02704792',
#  'n02177972', 'n04501370', 'n04026417', 'n02988304', 'n04380533', 'n02110341', 'n10565667', 'n02825657', 'n04467665',
#  'n03388549', 'n04548280', 'n04004767', 'n03633091', 'n01860187', 'n01910747', 'n02110063', 'n01914609', 'n03958227',
#  'n03884397', 'n02009912', 'n01945685', 'n01644373', 'n07583066', 'n02859443', 'n01968897', 'n04069434', 'n04487394',
#  'n07831146', 'n07880968', 'n04523525', 'n03710721', 'n02783161', 'n03535780', 'n03903868', 'n04254120', 'n03602883',
#  'n04252077', 'n01614925', 'n03026506', 'n02124075', 'n04418357', 'n02095889', 'n03134739', 'n03599486', 'n04019541',
#  'n01773797', 'n02107312', 'n02500267', 'n02667093', 'n03016953', 'n02116738', 'n03594945', 'n02504013', 'n03710637',
#  'n04254680', 'n04008634', 'n03255030', 'n02457408', 'n04462240', 'n02134418', 'n03717622', 'n04423845', 'n03770439',
#  'n03814639', 'n02841315', 'n02276258', 'n02012849', 'n03527444', 'n09399592', 'n02106030', 'n02111129', 'n01592084',
#  'n02514041', 'n04507155', 'n02123597', 'n03452741', 'n04532106', 'n07860988', 'n04355338', 'n01770081', 'n09468604',
#  'n01944390', 'n07892512', 'n04049303', 'n02808440', 'n02437312', 'n07716906', 'n03400231', 'n02090622', 'n03443371',
#  'n03126707', 'n04443257', 'n01697457', 'n03379051', 'n03179701', 'n03929660', 'n09288635', 'n02128925', 'n03459775',
#  'n03710193', 'n02104365', 'n04584207', 'n01784675', 'n01608432', 'n03759954', 'n03447721', 'n02823428', 'n04146614',
#  'n12144580', 'n02025239', 'n09256479', 'n03041632', 'n02092002', 'n02123045', 'n02793495', 'n04040759', 'n03207743',
#  'n01807496', 'n04131690', 'n02094433', 'n07802026', 'n02999410', 'n02346627', 'n02676566', 'n04263257', 'n03792972',
#  'n13052670', 'n01729977', 'n02002724', 'n02398521', 'n04465501', 'n02895154', 'n13054560', 'n03857828', 'n02094258',
#  'n02966193', 'n02110806', 'n04086273', 'n01688243', 'n01687978', 'n04118776', 'n03720891', 'n02807133', 'n02797295',
#  'n01751748', 'n03938244', 'n02114855', 'n03930313', 'n01855032', 'n02480855', 'n01981276', 'n04366367', 'n03355925',
#  'n02363005', 'n03532672', 'n03196217', 'n07717410', 'n04548362', 'n03534580', 'n04228054', 'n01773157', 'n02006656',
#  'n04127249', 'n04592741', 'n02481823', 'n03947888', 'n04428191', 'n04485082', 'n01770393', 'n02640242', 'n04612504',
#  'n03980874', 'n01796340', 'n04311004', 'n02992529', 'n07684084', 'n02835271', 'n02011460', 'n07734744', 'n03937543',
#  'n03100240', 'n03776460', 'n01882714', 'n03690938', 'n02093428', 'n03085013', 'n02236044', 'n02097474', 'n02814533',
#  'n02917067', 'n04552348', 'n06874185', 'n01871265', 'n04392985', 'n03775546', 'n03804744', 'n04273569', 'n04154565',
#  'n02782093', 'n01978287', 'n02871525', 'n03709823', 'n07718747', 'n02423022', 'n03868242', 'n01644900', 'n01819313',
#  'n01491361', 'n03666591', 'n02837789', 'n02002556', 'n03344393', 'n03924679', 'n03991062', 'n04579145', 'n03393912',
#  'n07579787', 'n02105855', 'n02190166', 'n09332890', 'n03916031', 'n02268443', 'n04328186', 'n02120079', 'n02233338',
#  'n03014705', 'n04252225', 'n02815834', 'n02102318', 'n04204347', 'n02395406', 'n02641379', 'n02093647', 'n04041544',
#  'n04357314', 'n02443484', 'n02112350', 'n02701002', 'n02488291', 'n02894605', 'n03388183', 'n03124170', 'n03345487',
#  'n04372370', 'n02098286', 'n01729322', 'n03930630', 'n07716358', 'n01682714', 'n02493509', 'n01986214', 'n02088466',
#  'n01740131', 'n01843065', 'n02087394', 'n04606251', 'n02091831', 'n01756291', 'n02165456', 'n02018207', 'n04409515',
#  'n03133878', 'n02342885', 'n04590129', 'n02110627', 'n03866082', 'n01534433', 'n02977058', 'n03877845', 'n04505470',
#  'n03249569', 'n02510455', 'n02483362', 'n02526121', 'n03840681', 'n04562935', 'n02389026', 'n02113023', 'n04325704',
#  'n06359193', 'n09193705', 'n02093754', 'n03670208', 'n02096585', 'n02017213', 'n03976657', 'n04493381', 'n01694178',
#  'n04254777', 'n03902125', 'n01514668', 'n04037443', 'n03110669', 'n04266014', 'n02979186', 'n02033041', 'n02109525',
#  'n03792782', 'n04389033', 'n03447447', 'n03691459', 'n02607072', 'n02096051', 'n02132136', 'n02108089', 'n02326432',
#  'n02509815', 'n03444034', 'n04509417', 'n04532670', 'n02109047', 'n02279972', 'n04344873', 'n03942813', 'n02093859',
#  'n02102973', 'n03445924', 'n04591157', 'n02100877', 'n03793489', 'n04447861', 'n02802426', 'n02167151', 'n12768682',
#  'n02091635', 'n02823750', 'n03297495', 'n01817953', 'n03485407', 'n02948072', 'n02095314', 'n02100236', 'n02066245',
#  'n04522168', 'n02883205', 'n01728920', 'n02086646', 'n03891332', 'n07754684', 'n01641577', 'n03018349', 'n03017168',
#  'n03530642', 'n04081281', 'n04542943', 'n01806567', 'n01748264', 'n03032252', 'n03095699', 'n02092339', 'n02981792',
#  'n04118538', 'n03899768', 'n02699494', 'n03482405', 'n04201297', 'n02808304', 'n04311174', 'n04599235', 'n04515003',
#  'n02093991', 'n04141975', 'n04525305', 'n03692522', 'n03478589', 'n01734418', 'n03000684', 'n03788365', 'n02445715',
#  'n02106382', 'n03761084', 'n02018795', 'n02129165', 'n09421951', 'n02117135', 'n04243546', 'n04483307', 'n03063599',
#  'n07745940', 'n03908618', 'n02749479', 'n02790996', 'n04417672', 'n03450230', 'n02113624', 'n04332243', 'n07930864',
#  'n02396427', 'n03089624', 'n02105056', 'n02840245', 'n04525038', 'n01531178', 'n03888257', 'n02422106', 'n04370456',
#  'n02119789', 'n04153751', 'n07873807', 'n03424325', 'n07836838', 'n03658185', 'n01530575', 'n02107908', 'n03314780',
#  'n02077923', 'n02536864', 'n01983481', 'n03223299', 'n03208938', 'n02107683', 'n04442312', 'n04550184', 'n07875152',
#  'n02085620', 'n03982430', 'n02091032', 'n04367480', 'n02966687', 'n02099849', 'n02051845', 'n02843684', 'n06596364',
#  'n01537544', 'n03131574', 'n03498962', 'n01704323', 'n07693725', 'n04275548', 'n02106166', 'n02869837', 'n03180011',
#  'n01737021', 'n15075141', 'n03494278', 'n02114712', 'n02483708', 'n13040303', 'n02690373', 'n03874293', 'n04486054',
#  'n01665541', 'n11939491', 'n09428293', 'n03673027', 'n02877765', 'n02095570', 'n02817516', 'n02494079', 'n03770679',
#  'n04152593', 'n03944341', 'n13133613', 'n04239074', 'n03476991', 'n02088632', 'n02443114', 'n02795169', 'n07871810',
#  'n02133161', 'n03662601', 'n04074963', 'n04560804', 'n02492035', 'n01774750', 'n09229709', 'n07717556', 'n02422699',
#  'n04204238', 'n02281406', 'n02978881', 'n02259212', 'n03803284', 'n02788148', 'n07248320', 'n02219486', 'n04044716',
#  'n04229816', 'n02408429', 'n01558993', 'n02087046', 'n03290653', 'n03733281', 'n01818515', 'n01978455', 'n04133789',
#  'n04355933', 'n02106550', 'n02492660', 'n02106662', 'n02113978', 'n02321529', 'n02088094', 'n04371774', 'n02951585',
#  'n02100583', 'n02071294', 'n03769881', 'n07753592', 'n04356056', 'n03207941', 'n03977966', 'n03970156', 'n02319095',
#  'n03676483', 'n02115913', 'n02490219', 'n01629819', 'n04596742', 'n01924916', 'n01632777', 'n02226429', 'n01443537',
#  'n03445777', 'n04398044', 'n03773504', 'n02096294', 'n04350905', 'n02013706', 'n03929855', 'n02281787', 'n09246464',
#  'n04479046', 'n02906734', 'n04536866', 'n02097047', 'n04597913', 'n02488702', 'n04208210', 'n04429376', 'n03124043',
#  'n03954731', 'n04120489', 'n03777568', 'n04209133', 'n03598930', 'n01675722', 'n03250847', 'n04554684', 'n01883070',
#  'n04277352', 'n02007558', 'n07760859', 'n02777292', 'n03272562', 'n01667778', 'n02115641', 'n04326547', 'n03742115',
#  'n03837869', 'n02102177', 'n04033995', 'n03680355', 'n02692877', 'n02317335', 'n02669723', 'n01917289', 'n03764736',
#  'n01518878', 'n02093256', 'n03721384', 'n03876231', 'n01582220', 'n03854065', 'n03623198', 'n03724870', 'n02056570',
#  'n02992211', 'n04149813', 'n02090379', 'n03495258', 'n01693334', 'n02747177', 'n02794156', 'n02356798', 'n02113186',
#  'n01692333', 'n04099969', 'n03483316', 'n02206856', 'n04604644', 'n02112137', 'n02172182', 'n12057211', 'n02786058',
#  'n03388043', 'n02097130', 'n02814860', 'n01797886', 'n03457902', 'n02101006', 'n04090263', 'n02486261', 'n02027492',
#  'n02280649', 'n03950228', 'n03394916', 'n03461385', 'n02094114', 'n03291819', 'n10148035', 'n03627232', 'n02113712',
#  'n02105412', 'n01616318', 'n01798484', 'n01749939', 'n04540053', 'n02127052', 'n04264628', 'n13044778', 'n07711569',
#  'n03240683', 'n02643566', 'n02037110', 'n03841143', 'n04404412', 'n02417914', 'n04330267', 'n01776313', 'n03661043',
#  'n02963159', 'n03995372', 'n02099712', 'n02120505', 'n04259630', 'n03992509', 'n02277742', 'n03220513', 'n04005630',
#  'n02174001', 'n01498041', 'n01669191', 'n03838899', 'n01755581', 'n02776631', 'n03146219', 'n02256656', 'n07715103',
#  'n02108551', 'n02441942', 'n03956157', 'n06785654', 'n01930112', 'n02865351', 'n07714990', 'n01855672', 'n02415577',
#  'n02493793', 'n02727426', 'n07747607', 'n02834397', 'n02091244', 'n04347754', 'n02101388', 'n07614500', 'n01496331',
#  'n02791270', 'n03983396', 'n02916936', 'n02391049', 'n01484850', 'n03347037', 'n03075370', 'n02870880', 'n02110958',
#  'n03877472', 'n01742172', 'n02090721', 'n03891251', 'n02804610', 'n07742313', 'n01985128', 'n03584829', 'n03782006',
#  'n02089867', 'n01689811', 'n09835506', 'n03372029', 'n01601694', 'n04009552', 'n01744401', 'n03920288', 'n01580077',
#  'n07768694', 'n03584254', 'n02397096', 'n02123159', 'n02447366', 'n12620546', 'n01440764', 'n04487081', 'n02268853',
#  'n06794110', 'n03888605', 'n03141823', 'n02951358', 'n02086910', 'n02125311', 'n01955084', 'n01768244', 'n03485794',
#  'n04458633', 'n04200800', 'n02454379', 'n01753488', 'n03733131', 'n03544143', 'n03425413', 'n02088364', 'n02879718',
#  'n07753113', 'n07584110', 'n01622779', 'n01990800', 'n02486410', 'n04589890', 'n03874599', 'n03590841', 'n03047690',
#  'n02484975', 'n02708093', 'n04162706', 'n07590611', 'n03743016', 'n02787622', 'n04179913', 'n03417042', 'n01824575',
#  'n03657121', 'n02487347', 'n02091134', 'n03272010', 'n03218198', 'n03785016', 'n03843555', 'n01774384', 'n03729826',
#  'n03786901', 'n01775062', 'n02111889', 'n04613696', 'n02113799', 'n03594734', 'n02412080', 'n02480495', 'n01514859',
#  'n01494475', 'n02489166', 'n03404251', 'n03649909', 'n02909870', 'n11879895', 'n04371430', 'n03935335', 'n03706229',
#  'n03733805', 'n03197337', 'n12267677', 'n04317175', 'n03337140', 'n03796401', 'n02085936', 'n01795545', 'n02100735',
#  'n03832673', 'n03781244', 'n02138441', 'n02930766', 'n02965783', 'n04136333', 'n01560419', 'n04235860', 'n07615774',
#  'n04067472', 'n04147183', 'n09472597', 'n02097209', 'n03271574', 'n02074367', 'n02028035', 'n03775071', 'n02442845',
#  'n02892767', 'n03216828', 'n02096437', 'n01833805', 'n03201208', 'n01664065', 'n04346328', 'n07697313', 'n02165105',
#  'n02769748', 'n02089973', 'n03998194', 'n02108000', 'n02939185', 'n03384352', 'n02606052', 'n02112018', 'n04335435',
#  'n04310018', 'n02971356', 'n01630670', 'n04111531', 'n03063689', 'n02105162', 'n02101556', 'n01739381', 'n02134084',
#  'n02108915', 'n03933933', 'n07714571', 'n02666196', 'n02497673', 'n02137549', 'n04461696', 'n04141076', 'n03045698',
#  'n02114367', 'n02730930', 'n02950826', 'n03777754', 'n04435653', 'n01980166', 'n03492542', 'n03000247', 'n02910353',
#  'n03788195', 'n04125021', 'n07718472', 'n04456115', 'n02085782', 'n03787032', 'n02097298', 'n07613480', 'n12985857',
#  'n03961711', 'n02504458', 'n04517823', 'n03976467', 'n02107142', 'n01735189', 'n02086240', 'n02129604', 'n01873310',
#  'n02058221', 'n02088238', 'n02009229', 'n03028079', 'n02099601', 'n03481172', 'n02110185', 'n02099267', 'n03791053',
#  'n04336792', 'n02325366', 'n02105641', 'n03887697', 'n03467068', 'n04141327', 'n04579432', 'n02169497', 'n01677366',
#  'n02098413', 'n04065272', 'n07920052', 'n03127747', 'n12998815', 'n02099429', 'n01667114', 'n01632458', 'n03062245',
#  'n04399382']

#
# origin_classes = np.load('origin_classes.npy').tolist()
# origin_classes.sort()
# print(np.array(origin_classes))
# mkdir('single_val_zn')
# for item in os.listdir('single_val'):
#     # mkdir()
#     # shutil.copy('single_val/%s' % item, 'single_val_zn/%s' % item)
#     if not item in origin_classes:
#         os.remove('single_val/%s' % item)

# print(np.load('top_weight_n03325584-42-2.npy.npy')[:,0])
#
# if 813 in np.load('top_weight_n03325584-42-2.npy.npy')[:,0]:
#     print(1111)


# print(np.load('n03325584-6-2.npy').shape)
# print(np.load('n03325584-7-2.npy').shape)

# print(np.load('n01491361-0-2.npy')[50:100].tolist())
# print(np.load('n01491361-0-2w.npy')[50:100].tolist())
#
# print(np.load('n04039381-0-2.npy')[50:100].tolist())
# print(np.load('n04039381-0-2_w.npy')[50:100].tolist())
#
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [0.71, 0.38, 0.58, 0.71, 0.44, 0.64, 0.39, 0.55, 0.49, 0.90]
y2 = [0.82, 0.38, 0.61, 0.79, 0.57, 0.72, 0.47, 0.68, 0.58, 0.90]

fig = plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.plot(x, y, linestyle='--', color='blue', label='修剪前')
plt.plot(x, y2, linestyle='-', color='black', label='修剪后')
plt.xlabel('类标号', fontsize=21)
plt.ylabel('召回率', fontsize=21)
plt.yticks(fontproperties='Times New Roman', size=18)
plt.xticks(fontproperties='Times New Roman', size=18)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.legend(prop={'size': 21})
plt.show()


# a = [[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],[1,2,3]],[[1,2,3],[1,2,3],9]]
# # a= np.array(a)
# print(a)