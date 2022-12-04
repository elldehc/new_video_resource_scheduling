import pandas as pd
import json

import CreateCameraApp
import UtilityFunction
import time
import ParetoOptimal
import sys

RES_MAP = {'360': '0', '600': '1', '720': '2', '900':'3', '1080':'4'}
FPS_MAP = {0: 2, 1: 3, 2: 5, 3: 10, 4: 15}
Interval_time = 10

def excel_to_dict_cloud(file_name):
    '''
    逐个读取excel中每个sheet，将其转化为dict
    :param file_name:
    :return: excel保存为dict形式返回，
                dict格式 {'object name': {config: profile_data, ..., config: profile_data}}
             bw 带宽占用
             edge_it edge推断时间
             cloud_it cloud推断时间
             edge_cu edge计算占用
             cloud_cu cloud计算占用
             ac: 准确度
    '''
    res_dict = {}
    car_dict = {}
    orginal_config = []
    df = pd.read_excel(file_name, sheet_name='Car')
    for i in df.index.values:
        config = df.loc[i][0]
        config = str(int(config)).zfill(3)
        orginal_config.append(config)
        # print(config)
        df_line = df.loc[i, ['bw', 'edge_it', 'cloud_it','edge_cu', 'cloud_cu', 'ac']].to_dict()
        # 将每一行转换成字典后添加到列表
        car_dict.update({config : df_line})
        # print(df_line)
    res_dict.update({'car':car_dict})
    pes_dict = {}
    df = pd.read_excel(file_name, sheet_name='Pes')
    for i in df.index.values:
        config = df.loc[i][0]
        config = str(int(config)).zfill(3)
        df_line = df.loc[i, ['bw', 'edge_it', 'cloud_it', 'edge_cu', 'cloud_cu', 'ac']].to_dict()
        # 将每一行转换成字典后添加到列表
        pes_dict.update({config: df_line})
        # print(df_line)
    # print(pes_dict)
    res_dict.update({'pes': pes_dict})
    return res_dict, orginal_config

def excel_to_dict_edge(file_name):
    '''
    逐个读取excel中每个sheet，将其转化为dict
    :param file_name:
    :return: excel保存为dict形式返回，
                dict格式 {'object name': {config: profile_data, ..., config: profile_data}}
             bw 带宽占用
             edge_it edge推断时间
             cloud_it cloud推断时间
             edge_cu edge计算占用
             cloud_cu cloud计算占用
             ac: 准确度
    '''
    res_dict = {}
    car_dict = {}
    orginal_config = []
    df = pd.read_excel(file_name, sheet_name='Car')
    for i in df.index.values:
        config = df.loc[i][0]
        config = str(int(config)).zfill(2)
        orginal_config.append(config)
        # print(config)
        df_line = df.loc[i, ['bw', 'edge_it', 'cloud_it','edge_cu', 'cloud_cu', 'ac']].to_dict()
        # 将每一行转换成字典后添加到列表
        car_dict.update({config : df_line})
        # print(df_line)
    res_dict.update({'car':car_dict})
    pes_dict = {}
    df = pd.read_excel(file_name, sheet_name='Pes')
    for i in df.index.values:
        config = df.loc[i][0]
        config = str(int(config)).zfill(2)
        df_line = df.loc[i, ['bw', 'edge_it', 'cloud_it', 'edge_cu', 'cloud_cu', 'ac']].to_dict()
        # 将每一行转换成字典后添加到列表
        pes_dict.update({config: df_line})
        # print(df_line)
    # print(pes_dict)
    res_dict.update({'pes': pes_dict})
    return res_dict, orginal_config

def read_json(filename):
    file = open(filename, 'r')
    string_camera_info = file.read()
    camera_info = json.loads(string_camera_info)
    file.close()
    # print(camera_info)
    return camera_info

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
def InitItem_start(edge_config_dict, cloud_config_dict, camera_app, edge_config, cloud_config,
                   MAX_BW, TRANS_TIME, MAX_INTER_MIGRATE, camera_maping):
    '''
    :return init_items 格式 {'camera0000': {edgeconfig&cloudconfig}: [edge_cu, cloud_cu, bw, sumutility,
                        [placement_edge_flag, placement_edge_flag], {appid: [placement, migration_flag]}]}
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
                for aid in camera_app[cid]:  # 当前camera下的每个app
                    # print(camera_app[cid][aid])
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
                    edge_U = UtilityFunction.utility(edge_accuracy, edge_latency, funID, preferID)
                    cloud_U = UtilityFunction.utility(cloud_accuracy, cloud_latency, funID, preferID)
                    # print(edge_U, cloud_U)
                    U = 0
                    if edge_U > cloud_U:
                        placement_flag_edge = 1
                        U = edge_U
                        app_placement.update({aid : [0, MAX_INTER_MIGRATE]})
                    else:
                        placement_flag_cloud = 1
                        U = cloud_U
                        app_placement.update({aid : [1, MAX_INTER_MIGRATE]})
                    per_app_utility.append(U)
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
                usage_edge_cu = [0 for i in range(region)]
                usage_edge_cu[cid_map_region] = edge_computing_usage
                usage_bw = [0 for i in range(region)]
                usage_bw[cid_map_region] = bandwidth_usage
                # print(str(placement_flag_edge)+ec+str(placement_flag_cloud)+cc)
                per_camera_items.update({str(placement_flag_edge)+ec+str(placement_flag_cloud)+cc: [usage_edge_cu, cloud_computing_usage,
                                                         usage_bw, sumU, per_app_utility,
                                                         [placement_flag_edge, placement_flag_cloud],
                                                         app_placement]})
        camera_items.update({cid: per_camera_items})
        # UtilityFunction.utility()
    # file = open('tmp_camera_app.json', 'w')
    # file.write(json.dumps(camera_items, indent=4))
    # file.close()
    # print(len(camera_items['camera0000']))
    return camera_items


def InitItem_next(pre_solution, edge_config_dict, cloud_config_dict, camera_app,
                                          edge_config, cloud_config,
                                          MAX_BW, TRANS_TIME, MAX_INTER_MIGRATE, camera_maping):
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
                usage_edge_cu = [0 for i in range(region)]

                usage_edge_cu[cid_map_region] = edge_computing_usage
                usage_bw = [0 for i in range(region)]
                usage_bw[cid_map_region] = bandwidth_usage
                per_camera_items.update({ec + '&' + cc: [usage_edge_cu, cloud_computing_usage,
                                                         usage_bw, sumU, per_app_utility,
                                                         [placement_flag_edge, placement_flag_cloud],
                                                         app_placement]})
        camera_items.update({cid: per_camera_items})
        # UtilityFunction.utility()
    # file = open('tmp_camera_app.json', 'w')
    # file.write(json.dumps(camera_items, indent=4))
    # file.close()
    # print(len(camera_items['camera0000']))
    return camera_items

def InitItems(pre_solution, edge_config_dict, cloud_config_dict, camera_app, edge_config, cloud_config,
              MAX_BW, TRANS_TIME, MAX_INTER_MIGRATE, camera_maping):
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
                                          MAX_BW, TRANS_TIME, MAX_INTER_MIGRATE, camera_maping)
        # print(init_iterm)
    else:
        print('current solution is not empty!!!')
        init_iterm = InitItem_next(pre_solution, edge_config_dict, cloud_config_dict, camera_app,
                                          edge_config, cloud_config,
                                          MAX_BW, TRANS_TIME, MAX_INTER_MIGRATE, camera_maping)
    return init_iterm

