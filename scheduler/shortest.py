import sys
import time

import CreateCameraApp
import ParetoOptimal
import numpy as np
from utils import *
import pickle

'''
solution输出格式：
{camera 1 : {'resource':[edge_cu, cloud_cu, bw], 'utility': U, 'loc' : [0,1] or [1, 0] or [1, 1],  
appid: [config, location, migration_flag],..., {appid: [config, location, migration_flag]}, 
'resource': resource, 'utility': utility, 'loc': loc, 'config': config}

camera 应用：
{cameraid: {appid: [utilty_function, objectid],...},..., cameraid} : 每个cameraid对应一个dict, 
dict保存app信息，每个appid包含使用的utility function和关注的待检测的objectid

objectid 0: car 1: pes

'''
'''
新应用的抵达，burst持续10-300s之间，拥有较高的quality和较低的延迟要求
网络带宽发生了变化 
'''

'''
solution输出格式：
{camera 1 : {'resource':[edge_cu, cloud_cu, bw], 'utility': U, 'loc' : [0,1] or [1, 0] or [1, 1],  
appid: [config, location, migration_flag],..., {appid: [config, location, migration_flag]}, 
'resource': resource, 'utility': utility, 'loc': loc, 'config': config}
'''



def InitItem_start(edge_config_dict, cloud_config_dict, camera_app, edge_config, cloud_config,
                   MAX_BW, TRANS_TIME, MAX_INTER_MIGRATE, camera_maping, app_num, alpha):
    '''
    :return init_items 格式 {'camera0000': {edgeconfig&cloudconfig}: [edge_cu, cloud_cu, bw, sumutility,
                        [placement_edge_flag, placement_edge_flag], {appid: [placement, migration_flag]}]}
    '''
    '''
    minu = sys.float_info.max
    maxu = sys.float_info.min
    print(camera_app)
    mm = {}
    for i in range(11):
        for j in range(11):
            mm[(0, 0, (i, j))] = [sys.float_info.max, sys.float_info.min]
            mm[(0, 1, (i, j))] = [sys.float_info.max, sys.float_info.min]
            mm[(0, 2, (i, j))] = [sys.float_info.max, sys.float_info.min]
            mm[(1, 0, (i, j))] = [sys.float_info.max, sys.float_info.min]
            mm[(1, 1, (i, j))] = [sys.float_info.max, sys.float_info.min]
            mm[(1, 2, (i, j))] = [sys.float_info.max, sys.float_info.min]
    '''
    camera_items = {}
    region = len(camera_maping)
    for cid in camera_app:  # 每个camera
        # print(cid)
        cid_map_region = sys.maxsize
        for i in camera_maping:
            if cid in camera_maping[i]:
                cid_map_region = i
        per_camera_items = {}

        for ec in edge_config:  # edge端的每个configuration
            # print(ec, type(ec))
            # edge_fps = int(FPS_MAP[ec.split(' ')[-1]])
            edge_fps = FPS_MAP[int(ec[-1])]
            for cc in cloud_config:  # cloud端的每个configuration

                # cloud_fps =int(FPS_MAP[cc.split(' ')[-2]])
                cloud_fps = FPS_MAP[int(cc[-2])]
                sumU = 0.0
                app_placement = {}
                placement_flag_edge = 0
                # 用于辨别当前视频流选择目标检测的位置，如果在edge执行，那么placement_flag_edge置1
                placement_flag_cloud = 0
                # 用于辨别当前视频流选择目标检测的位置，如果在cloud执行，那么placement_flag_cloud置1
                per_app_utility = []
                per_app_acc = []
                per_app_lat = []
                per_app_place = []
                for aid in camera_app[cid]:  # 当前camera下的每个app
                    # print(camera_app[cid][aid])
                    # print(aid)
                    appid = int(aid)
                    if appid >= app_num:
                        continue
                    app_alpha = alpha[appid]
                    objectID = camera_app[cid][aid][2]
                    funID = camera_app[cid][aid][0]
                    preferID = camera_app[cid][aid][1]
                    if objectID == 0:
                        object = 'car'
                    if objectID == 1:
                        object = 'pes'
                    edge_frame_data_size = edge_config_dict[object][ec]['bw'] / edge_fps
                    edge_latency = UtilityFunction.latency(edge_frame_data_size, MAX_BW,
                                                           edge_config_dict[object][ec]['edge_it'],
                                                           edge_config_dict[object][ec]['cloud_it'],
                                                           TRANS_TIME, 'edge')
                    cloud_frame_data_size = cloud_config_dict[object][cc]['bw'] / cloud_fps
                    cloud_latency = UtilityFunction.latency(cloud_frame_data_size, MAX_BW,
                                                            cloud_config_dict[object][cc]['edge_it'],
                                                            cloud_config_dict[object][cc]['cloud_it'],
                                                            TRANS_TIME, 'cloud')
                    # print(edge_latency, cloud_latency)
                    edge_accuracy = edge_config_dict[object][ec]['ac']
                    cloud_accuracy = cloud_config_dict[object][cc]['ac']
                    # print(edge_accuracy, cloud_accuracy)

                    edge_U = UtilityFunction.utility(edge_accuracy, edge_latency, funID, preferID, objectID, app_alpha)
                    cloud_U = UtilityFunction.utility(cloud_accuracy, cloud_latency, funID, preferID, objectID, app_alpha)

                    '''
                    if edge_U < mm[(objectID, preferID, app_alpha)][0]:
                        mm[(objectID, preferID, app_alpha)][0] = edge_U
                    if edge_U > mm[(objectID, preferID, app_alpha)][1]:
                        mm[(objectID, preferID, app_alpha)][1] = edge_U
                    if cloud_U < mm[(objectID, preferID, app_alpha)][0]:
                        mm[(objectID, preferID, app_alpha)][0] = cloud_U
                    if cloud_U > mm[(objectID, preferID, app_alpha)][1]:
                        mm[(objectID, preferID, app_alpha)][1] = cloud_U
                    # print(edge_U, cloud_U)
                    '''
                    U = 0

                    if edge_U > cloud_U:
                        placement_flag_edge = 1
                        U = edge_U
                        per_app_utility.append(U)
                        per_app_acc.append(edge_accuracy)
                        per_app_lat.append(edge_latency)
                        per_app_place.append(0)
                    else:
                        placement_flag_cloud = 1
                        U = cloud_U
                        per_app_utility.append(U)
                        per_app_acc.append(cloud_accuracy)
                        per_app_lat.append(cloud_latency)
                        per_app_place.append(1)
                    sumU = sumU + U
                # print(app_placement)
                # print(sumU)
                edge_computing_usage = 0
                cloud_computing_usage = 0
                bandwidth_usage = 0
                if  placement_flag_cloud+placement_flag_edge == 2:
                    edge_computing_usage = edge_config_dict[object][ec]['edge_cu'] + \
                                           cloud_config_dict[object][cc]['edge_cu']
                    cloud_computing_usage = cloud_config_dict[object][cc]['cloud_cu']
                    bandwidth_usage = cloud_config_dict[object][cc]['bw']
                elif placement_flag_cloud == 1:
                    edge_computing_usage = cloud_config_dict[object][cc]['edge_cu']
                    cloud_computing_usage = cloud_config_dict[object][cc]['cloud_cu']
                    bandwidth_usage = cloud_config_dict[object][cc]['bw']
                elif placement_flag_edge == 1:
                    edge_computing_usage = edge_config_dict[object][ec]['edge_cu']
                    # cloud_computing_usage = cloud_config_dict[object][cc]['cloud_cu']
                    # bandwidth_usage = cloud_config_dict[object][cc]['bw']

                profile = [0 for i in range(region*2+2)]
                # print(profile)
                # usage_edge_cu = [0 for i in range(region)]
                profile[cid_map_region] = edge_computing_usage
                profile[region] = cloud_computing_usage
                # usage_bw = [0 for i in range(region)]
                profile[region+cid_map_region+1] = bandwidth_usage
                profile[-1] = sumU
                # print(profile)

                per_camera_items.update({str(placement_flag_edge)+ec+str(placement_flag_cloud)+cc:
                                             [profile, [placement_flag_edge, placement_flag_cloud], per_app_utility,
                                              per_app_acc, per_app_lat, per_app_place]})
        camera_items.update({cid: per_camera_items})
        # UtilityFunction.utility()
    # file = open('tmp_camera_app.json', 'w')
    # file.write(json.dumps(camera_items, indent=4))
    # file.close()
    # print(len(camera_items['camera0000']))
    # print('max, min', mm)
    return camera_items


def InitItem_next(pre_solution, edge_config_dict, cloud_config_dict, camera_app,
                                          edge_config, cloud_config,
                                          MAX_BW, TRANS_TIME, MAX_INTER_MIGRATE, camera_maping, app_num):
    camera_items = {}
    region = len(camera_maping)
    for cid in camera_app:  # 每个camera
        # print(cid)
        cid_map_region = sys.maxsize
        for i in camera_maping:
            if cid in camera_maping[i]:
                cid_map_region = i
        per_camera_items = {}
        for ec in edge_config:  # edge端的每个configuration
            edge_fps = int(ec.split(' ')[-1])
            for cc in cloud_config:  # cloud端的每个configuration
                cloud_fps = int(cc.split(' ')[-2])
                sumU = 0.0
                app_placement = {}
                placement_flag_edge = 0
                # 用于辨别当前视频流选择目标检测的位置，如果在edge执行，那么placement_flag_edge置1
                placement_flag_cloud = 0
                # 用于辨别当前视频流选择目标检测的位置，如果在cloud执行，那么placement_flag_cloud置1
                per_app_utility = []
                for aid in camera_app[cid]:  # 当前camera下的每个app
                    # print(camera_app[cid][aid])
                    objectID = camera_app[cid][aid][2]
                    funID = camera_app[cid][aid][0]
                    preferID = camera_app[cid][aid][1]
                    pre_config = pre_solution[cid][aid][0] # 上一轮解决方案的 app的configuration
                    pre_loc = pre_solution[cid][aid][1]   # 上一轮解决方案的 app的location
                    pre_mig = pre_solution[cid][aid][2]
                    pre_only_cloud = 0
                    pre_only_edge = 0
                    pre_both = 0
                    if pre_solution[cid]['loc'] == [0, 1]:
                        pre_only_cloud = 1
                    if pre_solution[cid]['loc'] == [1, 0]:
                        pre_only_edge = 1
                    if pre_solution[cid]['loc'] == [1, 1]:
                        pre_both = 1
                    if objectID == 0:
                        object = 'car'
                    if objectID == 1:
                        object = 'pes'

                    edge_frame_data_size = edge_config_dict[object][ec]['bw'] / edge_fps
                    edge_latency = UtilityFunction.latency(edge_frame_data_size, MAX_BW,
                                                           edge_config_dict[object][ec]['edge_it'],
                                                           edge_config_dict[object][ec]['cloud_it'],
                                                           TRANS_TIME, 'edge')
                    cloud_frame_data_size = cloud_config_dict[object][cc]['bw'] / cloud_fps
                    cloud_latency = UtilityFunction.latency(cloud_frame_data_size, MAX_BW,
                                                            cloud_config_dict[object][cc]['edge_it'],
                                                            cloud_config_dict[object][cc]['cloud_it'],
                                                            TRANS_TIME, 'cloud')
                    # print(edge_latency, cloud_latency)
                    edge_accuracy = edge_config_dict[object][ec]['ac']
                    cloud_accuracy = cloud_config_dict[object][cc]['ac']
                    # print(edge_accuracy, cloud_accuracy)
                    edge_U = UtilityFunction.utility(edge_accuracy, edge_latency, funID, preferID)
                    cloud_U = UtilityFunction.utility(cloud_accuracy, cloud_latency, funID, preferID)
                    U = 0
                    # 从这里开始是对migration进行的处理，后续review时候需要注意，
                    # 这里只考虑了only-cloud到edge的迁移情况。
                    # 由于DNN partition的存在，tracking就只能在edge上进行，
                    # 这是不会存在stateful or stateless的情况，
                    # cloud必须将推断结果通过网络的形式返回给edge，
                    if pre_mig == 0:

                        if pre_loc == 0:
                            U = edge_U
                            placement_flag_edge = 1
                            app_placement.update({aid: [0, MAX_INTER_MIGRATE]})
                        else:
                            U = cloud_U
                            placement_flag_cloud = 1
                            app_placement.update({aid: [1, MAX_INTER_MIGRATE]})
                    else:

                        # cloud->edge migration cost = netlatency+migration_time

                        if edge_U > cloud_U & pre_only_cloud == 1:
                            migration_time = cloud_frame_data_size / MAX_BW + TRANS_TIME + \
                                             TRANS_TIME + 0.01831 / MAX_BW
                            frame_num = migration_time / edge_fps
                            total_frame_num = Interval_time * edge_fps

                            delta_edge_time = frame_num*edge_config_dict[object][ec]['edge_it']/\
                                              total_frame_num   # 补全滞后的帧，影响latency的情况
                            migration_edge_accuracy_U = UtilityFunction.utility(edge_accuracy,
                                                                         edge_latency+delta_edge_time,
                                                                         funID, preferID)
                            migration_edge_latency_U = edge_U*(Interval_time-migration_time)\
                                                       /Interval_time
                            max_U = max(migration_edge_accuracy_U, migration_edge_latency_U)

                            if max_U > cloud_U:
                                placement_flag_edge = 1
                                U = max_U
                                app_placement.update({aid: [0, pre_mig - 1]})
                            else:
                                placement_flag_cloud = 1
                                U = cloud_U
                                app_placement.update({aid: [1, pre_mig]})
                        elif edge_U > cloud_U:
                            placement_flag_edge = 1
                            U = edge_U
                            if pre_loc == 0:
                                app_placement.update({aid: [0, pre_mig]})
                            else:
                                app_placement.update({aid: [0, pre_mig - 1]})
                        elif edge_U <= cloud_U:
                            placement_flag_cloud = 1
                            U = cloud_U
                            if pre_loc == 0:
                                app_placement.update({aid: [1, pre_config-1]})
                            else:
                                app_placement.update({aid: [1, pre_config]})
                    sumU = sumU + U
                    per_app_utility.append(U)
                # print(app_placement)
                # print(sumU)
                edge_computing_usage = 0
                cloud_computing_usage = 0
                bandwidth_usage = 0
                if placement_flag_cloud + placement_flag_edge == 2:
                    edge_computing_usage = edge_config_dict[object][ec]['edge_cu'] + \
                                           cloud_config_dict[object][cc]['edge_cu']
                    cloud_computing_usage = cloud_config_dict[object][cc]['cloud_cu']
                    bandwidth_usage = cloud_config_dict[object][cc]['bw']
                elif placement_flag_cloud == 1:
                    edge_computing_usage = cloud_config_dict[object][cc]['edge_cu']
                    cloud_computing_usage = cloud_config_dict[object][cc]['cloud_cu']
                    bandwidth_usage = cloud_config_dict[object][cc]['bw']
                elif placement_flag_edge == 1:
                    edge_computing_usage = edge_config_dict[object][ec]['edge_cu']
                    # cloud_computing_usage = cloud_config_dict[object][cc]['cloud_cu']
                    # bandwidth_usage = cloud_config_dict[object][cc]['bw']
                profile = [0 for i in range(region*2+2)]
                # print(profile)
                # usage_edge_cu = [0 for i in range(region)]
                profile[cid_map_region] = edge_computing_usage
                profile[region] = cloud_computing_usage
                # usage_bw = [0 for i in range(region)]
                profile[region + cid_map_region + 1] = bandwidth_usage
                profile[-1] = sumU

                per_camera_items.update({str(placement_flag_edge)+ec+str(placement_flag_cloud)+cc:
                                             [profile , [placement_flag_edge, placement_flag_cloud]]})
        camera_items.update({cid: per_camera_items})
        # UtilityFunction.utility()
    # file = open('tmp_camera_app.json', 'w')
    # file.write(json.dumps(camera_items, indent=4))
    # file.close()
    # print(len(camera_items['camera0000']))
    return camera_items

def InitItems(pre_solution, edge_config_dict, cloud_config_dict, camera_app, edge_config, cloud_config,
              MAX_BW, TRANS_TIME, MAX_INTER_MIGRATE, camera_maping, app_num, alpha):
    '''
        :return init_items 格式 {'camera0000':
                                    {edgeconfig&cloudconfig}: [edge_cu, cloud_cu, bw, sumutility,
                                    [placement_edge_flag, placement_edge_flag],
                                    {appid: [placement, migration_flag]}]}
    '''
    if not pre_solution:
        print('current solution is empty!!!')
        init_iterm = InitItem_start(edge_config_dict, cloud_config_dict, camera_app,
                                          edge_config, cloud_config,
                                          MAX_BW, TRANS_TIME, MAX_INTER_MIGRATE, camera_maping, app_num, alpha)
        # print(init_iterm)
    else:
        print('current solution is not empty!!!')
        init_iterm = InitItem_next(pre_solution, edge_config_dict, cloud_config_dict, camera_app,
                                          edge_config, cloud_config,
                                          MAX_BW, TRANS_TIME, MAX_INTER_MIGRATE, camera_maping, app_num, alpha)
    return init_iterm




def find_lowest_solution(init_items, region_num):
    current_solution = {}
    # total = np.zeros(region_num*2+2)
    # print('qqqqq', init_items['0000']['00'])
    total = [0 for i in range(region_num*2+2)]
    for camera_id in init_items:
        # print(camera_id)
        min_utility = sys.float_info.max
        items_info = []
        min_config = None
        current_camera_solution = {}
        # print(init_items[camera_id]['00'])
        for config in init_items[camera_id]:
            # print(init_items[camera_id][config])
            if init_items[camera_id][config][0][-1] < min_utility:
                min_utility = init_items[camera_id][config][0][-1]
                items_info = init_items[camera_id][config]
                min_config = config

        # print(min_utility, items_info)
        resource_usage = items_info[0][0:-1]
        utility = items_info[0][-1]

        current_camera_solution.update({'config': min_config})
        current_camera_solution.update({'resource': resource_usage})
        current_camera_solution.update({'utility': utility})
        current_camera_solution.update({'appu': items_info[2]})
        current_camera_solution.update({'appac': items_info[3]})
        current_camera_solution.update({'applat': items_info[4]})
        current_camera_solution.update({'place': items_info[5]})
        # current_camera_solution.update({'loc': loc})
        for i in range(region_num*2+2):
            total[i] = total[i] + items_info[0][i]
        current_solution.update({camera_id: current_camera_solution})
        # print(current_camera_solution)
        # print(current_solution)
        # break
    current_solution.update({'resource': total[0:-1]})
    current_solution.update({'utility': total[-1]})
    # current_solution.update({'cloud_usage': total_cloud_usage})
    # current_solution.update({'bw_usage': total_bw})
    # current_solution.update({'utility': total_utility})
    '''
    for key in current_solution:
        if 'camera' in key:
            tmp = ['360 2']
            if current_solution[key]['config'] != tmp:
                print(key, current_solution[key]['config'])
    '''
    # print(current_solution['edge_usage'])
    return current_solution

def PaneltyVecter(current_solution, R):
    pv1 = current_solution['resource']
    pv2 = []
    for i in range(len(R)):
        pv2.append(pv1[i]/(R[i]-pv1[i]))
    # print(pv1)

    return pv2


def FilterInitItems(init_items):
    filtered_items = {}
    for camera_id in init_items:
        # print(init_items[camera_id])
        camera_filtered_items = {}
        config_list = []
        for config in init_items[camera_id]:
            # print(init_items[camera_id][config])
            # print(config)
            edge_conf = config[1:3]
            # print(edge_conf)
            cloud_conf = config[4:]
            # print(edge_conf, cloud_conf)
            conf = None
            if init_items[camera_id][config][1] == [1, 0]:
                conf = edge_conf
            elif init_items[camera_id][config][1] == [0, 1]:
                conf = cloud_conf
            elif init_items[camera_id][config][1] == [1, 1]:
                conf = config
            if conf in config_list:
                continue
            else:
                config_list.append(conf)
                camera_filtered_items.update({conf: init_items[camera_id][config]})
        filtered_items.update({camera_id: camera_filtered_items})
    # file = open('tmp_camera_app.json', 'w')
    # file.write(json.dumps(filtered_items['0000'], indent=4))
    # file.close()
    return filtered_items

def break_point_match(resource_usage, R):
    # print(a, b,  c)
    # print(resource_usage)
    f = False
    for i in range(len(R)):
        if resource_usage[i] > R[i]:
            f = True
            break
    return f

def UpgradeUtility(current_solution, camera_replace, config_replace, filtered_items):

    # pre_resource_usage = np.array(current_solution[camera_replace]['resource'])
    pre_resource_usage = current_solution[camera_replace]['resource']

    # cur_resource_usage = np.array(filtered_items[camera_replace][config_replace][0][0:-1])
    cur_resource_usage = filtered_items[camera_replace][config_replace][0][0:-1]


    # pre_total_resource_usage = np.array(current_solution['resource'])
    pre_total_resource_usage = current_solution['resource']
    cur_total_resource_usage = []
    for i in range(len(cur_resource_usage)):
        cur_total_resource_usage.append(pre_total_resource_usage[i] + \
                                      cur_resource_usage[i] - pre_resource_usage[i])

    # cur_total_utility = pre_total_utility + cur_utility - pre_utility

    return cur_total_resource_usage


def MultidimensionResourceReduction(current_solution, pv, init_items,
                                    max_single_config, camera_maping,
                                    single_resource, angular_coefficient,
                                    region_num):
    # start_time = time.time()
    if max_single_config is None:
        for camera_id in init_items:
            # print(init_items[camera_id])
            for config in init_items[camera_id]:
                # print(config)
                # conf = (camera_id, config)
                if config is None:
                    continue
                # print(camera_id, config)
                conf = camera_id+config

                # print(conf)
                single_resource_usage = 0.0
                resource_weight = []
                # print(init_items[camera_id][config][0], type(init_items[camera_id][config][0]))
                for i in range(len(init_items[camera_id][config][0])-1):
                    resource_in_weight = init_items[camera_id][config][0][i] * pv[i]
                    single_resource_usage = single_resource_usage + resource_in_weight
                    resource_weight.append(resource_in_weight)
                # print(tmp.sum(), single_resource_usage)

                if single_resource_usage == 0.0:
                    single_resource_usage = sys.float_info.max
                '''
                if max_tmp < init_items[camera_id][config][3] / single_resource_usage:
                    max_config = conf
                    max_tmp = init_items[camera_id][config][3] / single_resource_usage
                '''
                angular_coefficient.update({conf: init_items[camera_id][config][0][-1] / single_resource_usage})
                single_resource.update({conf: [single_resource_usage, resource_weight]})
    else:   # 时间占用耗费点，这里需要优化
        camera_id = max_single_config[0]
        config = max_single_config[1]
        start_time = time.time()
        # print('1111', camera_id, len(config))
        # 判断上次更新配置更改的资源维度
        len_c = len(config)
        edge = 1
        cloud = 1
        if len_c == 3:
            edge = 0
        elif len_c == 2:
            cloud = 0
        region_change = 0
        for region in camera_maping:
            if camera_id in camera_maping[region]:
                region_change = region
                break
        # print(cloud)
        # assert edgepair[0][0:4]
        # print(init_items)
        # assert False
        if edge is 1:
            for camera_id in camera_maping[region_change]:
                # print(init_items[camera_id])
                for config in init_items[camera_id]:
                    conf = camera_id + config
                    # conf = (camera_id, config)
                    single_resource[conf][0] = single_resource[conf][0] - single_resource[conf][1][region_change]
                    single_resource[conf][1][region_change] = \
                        init_items[camera_id][config][0][region_change] * pv[region_change]
                    single_resource[conf][0] = single_resource[conf][0] + single_resource[conf][1][region_change]
                    angular_coefficient[conf] = init_items[camera_id][config][0][-1] / single_resource[conf][0]
        if cloud is 1:
            for camera_id in init_items:
                for config in init_items[camera_id]:
                    if len(config) == 2:
                        continue
                    # conf = (camera_id, config)
                    conf = camera_id+config
                    # print(single_resource[conf])
                    single_resource[conf][0] = single_resource[conf][0] - \
                                               single_resource[conf][1][region_num] - \
                                               single_resource[conf][1][region_num + region_change + 1]
                    single_resource[conf][1][region_num + region_change + 1] = \
                        init_items[camera_id][config][0][region_num + region_change + 1] * \
                        pv[region_num + region_change + 1]

                    single_resource[conf][1][region_num] = \
                        init_items[camera_id][config][0][region_num] * pv[region_num]

                    single_resource[conf][0] = single_resource[conf][0] + \
                                               single_resource[conf][1][region_num + region_change + 1] + \
                                               single_resource[conf][1][region_num]
                    angular_coefficient[conf] = init_items[camera_id][config][0][-1] / single_resource[conf][0]
    '''
        for camera_id in init_items:
            # print(init_items[camera_id])
            for config in init_items[camera_id]:
                conf = camera_id + config
                if edge is 1:
                    single_resource[conf][0] = single_resource[conf][0] - single_resource[conf][1][region_change]
                    single_resource[conf][1][region_change] = \
                        init_items[camera_id][config][0][region_change]*pv[0][region_change]
                    single_resource[conf][0] = single_resource[conf][0] + single_resource[conf][1][region_change]
                    angular_coefficient[conf] = init_items[camera_id][config][3] / single_resource[conf][0]
                if cloud is 1:
                    single_resource[conf][0] = single_resource[conf][0] - \
                                               single_resource[conf][1][region_num] - \
                                               single_resource[conf][1][region_num+region_change+1]
                    single_resource[conf][1][region_num+region_change+1] = \
                        init_items[camera_id][config][2][region_change] * pv[2][region_change]
                    single_resource[conf][1][region_num] = init_items[camera_id][config][1] * pv[1]

                    single_resource[conf][0] = single_resource[conf][0] + \
                                               single_resource[conf][1][region_num+region_change+1] + \
                                               single_resource[conf][1][region_num]
                    angular_coefficient[conf] = init_items[camera_id][config][3] / single_resource[conf][0]
        end_time = time.time()
        print(end_time - start_time)
        '''

    # end_time = time.time()
    # print(end_time - start_time)
    sorted_angular_coefficient = sorted(angular_coefficient.items(), key=lambda x: x[1], reverse=True)
    # rint(sorted_angular_coefficient[0], sorted_angular_coefficient[1])
    # print(angular_coefficient[0], angular_coefficient[1])

    # return sorted_angular_coefficient[0][0], single_resource, angular_coefficient
    flag = True
    return_config = None
    resource_usage = None
    for pair in sorted_angular_coefficient:
        # print(pair[0])
        # camera_id = pair[0][0]
        # config = pair[0][1]
        camera_id = pair[0][0:4]
        config = pair[0][4:]
        # print(camera_id, config)
        # print(init_items[camera_id][config][3])
        # print("111", current_solution[camera_id]['utility'], init_items[camera_id][config][0][-1])
        if current_solution[camera_id]['utility'] < init_items[camera_id][config][0][-1]:
            resource_usage = UpgradeUtility(current_solution, camera_id, config, init_items)

            if  break_point_match(resource_usage, R) == False:

                flag = False
                return_config = pair[0]
                break
    return current_solution, return_config, single_resource, angular_coefficient, flag, resource_usage



def UpgradeSolution(current_solution, single_resource, angular_coefficient,
                    max_c, filtered_items, resource_usage):
    # print(max_resource_angular_coefficient)
    # print(current_solution[camera_replace])
    # print(filtered_items[camera_replace])
    camera_replace = max_c[0:4]
    config_replace = max_c[4:]
    camera_replace_dict = {}

    camera_replace_dict.update({'config': config_replace})
    camera_replace_dict.update({'resource': filtered_items[camera_replace][config_replace][0][0:-1]})
    camera_replace_dict.update({'utility': filtered_items[camera_replace][config_replace][0][-1]})
    camera_replace_dict.update({'appu': filtered_items[camera_replace][config_replace][2]})
    camera_replace_dict.update({'appac': filtered_items[camera_replace][config_replace][3]})
    camera_replace_dict.update({'applat': filtered_items[camera_replace][config_replace][4]})
    camera_replace_dict.update({'place': filtered_items[camera_replace][config_replace][5]})
    # camera_replace_dict.update({'loc': filtered_items[camera_replace][config_replace][5]})
    # print(filtered_items[camera_replace][config_replace][6])
    # print(camera_replace_dict)
    pre_utility = current_solution[camera_replace]['utility']


    cur_utility = filtered_items[camera_replace][config_replace][0][-1]

    '''
    print(pre_edge_usage, cur_edge_usage)
    print(pre_cloud_usage, cur_cloud_usage)
    print(pre_bw_usage, cur_bw_usage)
    print(pre_utility, cur_utility)
    '''
    pre_total_utility = current_solution['utility']


    cur_total_utility = pre_total_utility + cur_utility - pre_utility
    '''
    print(pre_total_edge_usage, cur_total_edge_usage)
    print(pre_total_cloud_usagee, cur_total_cloud_usage)
    print(pre_total_bw_usagee, cur_total_bw_usage)
    print(pre_total_utility, cur_total_utility)
    '''
    current_solution[camera_replace] = camera_replace_dict
    current_solution['resource'] = resource_usage
    current_solution['utility'] = cur_total_utility
    filtered_items[camera_replace].pop(config_replace)
    single_resource.pop(max_c)
    angular_coefficient.pop(max_c)

    return current_solution, single_resource, angular_coefficient, filtered_items
    # 'edge_usage': [36.15000000000001, 38.32500000000001,
    # 40.50000000000001, 42.29500000000001, 36.15000000000001],
    # 'cloud_usage': 0.0, 'bw_usage': [0.0, 0.0, 0.0, 0.0, 0.0], 'utility': 69.24059999999999}





if __name__ == '__main__':
    edge_excel_file = 'edge_profile_result.xlsx'
    cloud_excel_file = 'cloud_profile_result.xlsx'





    cudasift_time = 0.0021
    MAX_INTER_MIGRATE = 1  # 表示最大迁移间隔 1代表隔一次才能进行迁移
    # MAX_CLOUD_GPU, MAX_EDGE_GPU, MAX_BW, TRANS_TIME, MAX_INTER_MIGRATE
    # R = [200, 200, 200, 200, 200, 1000, 210, 210, 210, 210, 210]
    # R = [200, 200, 200, 200, 200, 1000, 100, 100, 100, 100, 100]
    # R = [200, 200, 200, 200, 200, 1000, 150, 150, 150, 150, 150]
    # R = [200, 200, 200, 200, 200, 1000, 50, 50, 50, 50, 50]
    # R = [200, 200, 200, 200, 200, 1000, 300, 300, 300, 300, 300]

    # R = [100, 100, 100, 100, 100, 1000, 210, 210, 210, 210, 210]
    # R = [300, 300, 300, 300, 300, 1000, 210, 210, 210, 210, 210]
    # R = [400, 400, 400, 400, 400, 1000, 210, 210, 210, 210, 210]
    # R = [500, 500, 500, 500, 500, 1000, 210, 210, 210, 210, 210]

    # R = [200, 200, 200, 200, 200, 1000, 210, 210, 210, 210, 210]
    edge_config_dict_all, d = excel_to_dict_edge(edge_excel_file)  # 测量数据中未经过过滤的全部的configuration
    cloud_config_dict_all, c = excel_to_dict_cloud(cloud_excel_file)  # 测量数据中未经过过滤的全部的configuration
    # print(cloud_config_dict)
    pre_solution = {}
    edge_config_dict, cloud_config_dict, edge_config, cloud_config = \
        ParetoOptimal.ParetoFilter(edge_config_dict_all, cloud_config_dict_all)

    print(len(edge_config), len(cloud_config))
    # edge_config, cloud_config = ParetoOptimal.ConfigurationVideo(edge_config_dict, cloud_config_dict)
    # print(len(edge_config_dict['car']), len(cloud_config_dict['car']))

    def config_save_yaml(yaml_name, res):
        conf = []
        import yaml
        yamltxt = yaml.dump(res)
        fp =  open(yaml_name, 'w')
        fp.write(yamltxt)
        fp.close()
        '''
        for cid in res.keys():
            print(cid)
            if cid is 'resource' and cid is '0':
                continue
            else:
                
        '''

    def section(init_camera_json, R, MAX_BW, app_num, alpha, j):
        print(R)
        camera_info = read_json(init_camera_json)
        camera_maping = CreateCameraApp.CameraEdgeMaping(camera_info, region_num, len(camera_info))
        # print(camera_maping)
        start_time = time.time()
        # print(cloud_config)
        init_items = InitItems(pre_solution, edge_config_dict, cloud_config_dict, camera_info,
                               edge_config, cloud_config, MAX_BW, TRANS_TIME,
                               MAX_INTER_MIGRATE, camera_maping, app_num, alpha)
        # print(init_items['0001'])
        # assert False
        # print(init_items['0000']['1010'])


        filtered_items = FilterInitItems(init_items)
        # print(len(filtered_items['0000']))
        # print(filtered_items['0000'])
        end_time = time.time()
        # print('filtered items:', len(init_items['0000']))
        # print('InitItems:', end_time - start_time)
        start_t = time.time()
        current_solution = find_lowest_solution(filtered_items, region_num)
        # print(current_solution['resource'])
        # print(current_solution['utility'])
        ## print(current_solution)
        res = {}
        cnt = 0
        single_resource = {}
        angular_coefficient = {}
        max_resource_angular_coefficient = None
        while (1):
            # print(cnt)
            res = current_solution
            pv = PaneltyVecter(current_solution, R)
            start_time = time.time()
            current_solution, max_c, single_resource, angular_coefficient, flag, resource_usage = \
                MultidimensionResourceReduction(current_solution, pv, filtered_items,
                                                max_resource_angular_coefficient, camera_maping,
                                                single_resource, angular_coefficient, region_num)
            if flag:
                break
            end_time = time.time()
            # print('MultidimensionResourceReduction:', end_time - start_time)
            # print(max_resource_angular_coefficient)
            current_solution, single_resource, angular_coefficient, filtered_items = UpgradeSolution(current_solution,
                                                                                                     single_resource,
                                                                                                     angular_coefficient,
                                                                                                     max_c,
                                                                                                     filtered_items,
                                                                                                     resource_usage)

            # end_time = time.time()
            # print('UpgradeUtility:', end_time - start_time)
            # break
            cnt = cnt + 1
            # if cnt == 405:
            #     break
            # break
        end_t = time.time()
        # print(cnt)
        # print(res)
        print('utility: ', res['utility'])
        print('resource: ', res['resource'])
        print(res)
        tmp = []
        for i in range(len(res['resource'])):
            tmp.append(res['resource'][i] / R[i])
        print('min resource utilization: ', min(tmp))
        print('time:', end_t - start_t)
        application_u = []
        application_acc = []
        application_lat = []
        application_place = []
        count = 0
        cloud_c = 0
        p_c = 0
        # print(res['0000']['config'])
        for cid in res:
            if len(cid) is 4:
                for u in res[cid]['appu']:
                    application_u.append(u)
                for acc in res[cid]['appac']:
                    application_acc.append(acc)
                for lat in res[cid]['applat']:
                    application_lat.append(lat)
                for p in res[cid]['place']:
                    application_place.append(p)
                if res[cid]['config'] is None:
                    continue
                if len(res[cid]['config']) > 4:
                    count = count + 1
                if len(res[cid]['config']) >= 3:
                    cloud_c = cloud_c+1
                    if int(res[cid]['config'][-1]) > 0:
                        p_c = p_c+1
        # print(application_u)
        print(count)
        num_app = len(application_place)
        # print(num_app)
        u = (max(application_u), min(application_u), np.mean(application_u), application_place.count(0) / num_app,
            application_place.count(1) / num_app)
        a = (max(application_acc), min(application_acc), np.mean(application_acc),
               application_place.count(0) / num_app, application_place.count(1) / num_app)
        l = (max(application_lat), min(application_lat), np.mean(application_lat),
               application_place.count(0) / num_app, application_place.count(1) / num_app)

        yaml_name = 'sol_lw_{}.yaml'.format(j+1)

        config_save_yaml(yaml_name, res)

        return u, a, l, count, (cloud_c, p_c)

        # count是放置在云端的
        # print(res)

        # print(filtered_items['camera0037']['720 3 1'])

    
    # u, a, l = section(init_camera_json, R, 100, app_num)
    # print(u ,a, l)
    # init_camera_json = 'camera_app-{m,50,4}.json'
    init_camera_json = 'camera_app-{m,8,4}.json'
    accuracy = []
    latency = []
    utility = []
    # L_app = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    # L_app = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    cnt = []
    p = []
    TRANS_TIME = 0.12
    # region_num = 5
    alpha_l=pickle.load(open("alpha_l.pkl","rb"))
    region_num = 2  # edge分为5个片区，每个片区2个edge gpu 10个camera

    # R = [200, 200, 200, 200, 200, 800, 100, 100, 100, 100, 100]
    # json_name = ['camera_app-{m,50,1}.json', 'camera_app-{m,50,2}.json', 'camera_app-{m,50,3}.json', 'camera_app-{m,50,4}.json', 'camera_app-{m,50,5}.json']
    for j in range(len(alpha_l)):
        R = [100, 100, 200, 50, 50]
        u, a, l, count, pp = section(init_camera_json, R, 50, 24, alpha_l[j], j)
        accuracy.append(a[2])
        latency.append(l[2])
        utility.append(u[2])
        cnt.append(count)
        p.append(pp)

    print()
    print(accuracy)
    print(latency)
    print(utility)
    print(cnt)
    print(p)
    '''
    accuracy = []
    latency = []
    utility = []
    B = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

    # json_name = ['camera_app-{m,50,1}.json', 'camera_app-{m,50,2}.json', 'camera_app-{m,50,3}.json', 'camera_app-{m,50,3}.json', 'camera_app-{m,50,5}.json']
    for j in B:

        R = [200, 200, 200, 200, 200, 800, j, j, j, j, j]
        u, a, l = section(init_camera_json, R, j, app_num)
        accuracy.append(a)
        latency.append(l)
        utility.append(u[2])
    print()
    print(utility)
    '''
    # print(accuracy)
    # print(latency)



