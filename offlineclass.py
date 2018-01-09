# -*- coding: utf-8 -*-
from datetime import datetime
from datetime import timedelta
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def read_data(dataname):
    if dataname == 'item_all':
        return pd.read_csv('data/tianchi_fresh_comp_train_item.csv',
                           dtype={'item_id': str, 'item_geohash': str, 
                                  'item_catagory': str})
    if dataname == 'user_all':
        return pd.read_csv('data/tianchi_fresh_comp_train_user.csv',
                           dtype={'user_id': str, 'item_id': str, 
                                  'user_geohash': str,'behavior_type': str, 
                                  'item_catagory': str}, 
                           parse_dates=['time'])
#%%
def layer(layername, x, output_size=int, keep_prob=1.0, activation_function=None):
    # add one more layer and return the output of this layer  
    in_size=x.shape[1].value
    with tf.name_scope(layername):
        with tf.name_scope('Weights'):
            W = tf.Variable(tf.truncated_normal([in_size, output_size],
                            stddev=0.1))
            tf.summary.histogram('value', W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.truncated_normal([1, output_size],
                            stddev=0.1))
            tf.summary.histogram('value', b)
        with tf.name_scope('Wx_plus_b'):
            output = tf.nn.dropout(tf.matmul(x, W)+b, keep_prob)
            if activation_function:
                output = activation_function(output)
    tf.summary.histogram(layername+'/output', output)
    return output

#%%
def multilayer(n_layer, x, output_size, keep_prob, activation_function):
    for n in range(n_layer):
        layername = 'layer' + str(n+1)
        x = layer(layername, x, output_size=output_size[n], 
                  keep_prob=keep_prob[n], 
                  activation_function=activation_function[n])
    return x

#%%
########################## 数据调查的相关程序 ##########################
# a1：选取其中每个用户购买比较多的物品,没用到class User()
def a1(user_bought, answer=pd.DataFrame(columns=['user_id', 'item_id']), bottom_size=24):
    user_bought_id_list = pd.Series(user_bought['user_id'].unique())
    for user_id in user_bought_id_list:
        # 读取每个用户的信息，user_info为特定user_id的所有购买信息
        user_info = user_bought[user_bought['user_id'] == user_id]
        # 选取购买次数大于bottomsize的，而且其实可以不排序的，先这样吧
        user_info_size = user_info.groupby(['item_id']).size().sort_values(ascending=False)
        user_info_size_top = user_info_size[user_info_size > bottom_size].reset_index()
        user_info_size_top['user_id'] = user_id
        result = user_info_size_top[['user_id', 'item_id']]
        # 添加到answer这个df里去
        answer = pd.concat([answer, result])
    return answer

# a2：看看从点击到购买大概要多少时间(最近一次算了13个小时，还行吧),没用到class User()
def a2():
    user_bought_id_list = pd.Series(user_bought['user_id'].unique())
    # 建立空Series
    all_period = pd.Series()
    # 每个用户分别弄一下吧
    for user_id in user_bought_id_list:
        # user_info为特定user_id的所有信息，并按时间先后排序
        user_info = user_all[user_all['user_id'] == user_id]
        user_info.sort_values(by='time', inplace=True)
        # 提取买过东西的user_id和item_id
        user_bought_info = user_bought[user_bought['user_id'] == user_id]
        item_bought_list = user_bought_info['item_id'].unique()
        # 对每个item分别开始计算
        for item_id in item_bought_list:
            user_item_all = user_info[user_info['item_id'] == item_id]
            # 去掉重复的行
            user_item_all.drop_duplicates(inplace=True)
            # 只保留点击(1)和购买(4)的行为
            user_item_all = user_item_all.loc[(user_item_all['behavior_type'] == '1') |
                                             (user_item_all['behavior_type'] == '4')]
            # 获取购买行为的行序数
            user_item_all['count'] = range(1, (len(user_item_all)+1))
            # ↓仅为购买行为
            user_item_4_count = user_item_all.loc[user_item_all['behavior_type'] == '4', 'count']
            # 计算每次购买行为的时间间隔——点击~购买的时间间隔，并放入all_period
            count_start = 0
            for count_4 in user_item_4_count:
                # 去除只有购买没有点击的框（为了统计的逻辑合理）
                if count_4 == count_start + 1:
                    count_start = count_4
                    continue
                count_end = count_4
                temp = user_item_all[count_start:count_end]
                period = (temp['time'].values[-1]-temp['time'].values[0])
                all_period = all_period.append(pd.Series(period))
                count_start = count_end
    # 将all_period groupby一下，并存储
    all_period.groupby(0).size().sort_index().to_csv('data/a2-all_period.csv',
                                                     encoding='gbk')
    # 读取，并处理groupby之后的数据
    all_period_size = pd.read_csv('data/a2-all_period.csv', encoding='gbk',
                                  names=['time', 'size'])
    all_period_size['hours'] = all_period_size['time'].apply(
        lambda x: int(x.split(':')[0].split(' days ')[0]) * 24 +
                  int(x.split(':')[0].split(' days ')[1]))
    all_period_size['percent'] = all_period_size['size'] / all_period_size['size'].sum() * 100
    all_period_size.to_csv('data/a2-all_period_size.csv',
                                                     encoding='gbk')
    # 画图
    plt.figure
    plt.bar(all_period_size['hours'][:48], all_period_size['percent'][:48])
    plt.grid()
    plt.xlabel('Time(h)')
    plt.ylabel('percent')
    plt.title('percent')
    plt.show()
#########构建特征和答案#############
# 针对user_all的预处理
def u1(user_all):
    user_all['days'] = (user_all['time'] - datetime(2014, 11, 18)).apply(lambda x: x.days)
    user_all['hours'] = (user_all['time'] - datetime(2014, 11, 18, 0)).apply(lambda x: x.seconds // 3600)
    user_all_xy = user_all[['user_id', 'item_id', 'days', 'hours']].drop_duplicates()
    return user_all_xy, user_all

# y1：直接在user_all上给y赋值，应该是更快的
def y1(user_all_xy, backdays=-1):
    user_all_btype_4 = user_all.loc[user_all['behavior_type'] == '4',
                            ['user_id', 'item_id', 'days']].drop_duplicates()
    user_all_btype_4['days'] = user_all_btype_4['days'] + backdays
    user_all_btype_4['y'] = 1# 10654899个
    user_all_y = pd.merge(user_all_xy, user_all_btype_4, on=['user_id',
                            'item_id', 'days'], how='left').fillna({'y': 0})
    return user_all_y

# f1:构建特征1——单pair当天的操作（1，2，3，4）次数(至于重复操作问题，再单列去重的一条
def f1(user_all_xy, behavior_type='1', columns=[]):
    name = 'x1_' + behavior_type #特征名
    columns.append(name) #加入特征名
    user_all_btype = user_all.loc[user_all['behavior_type'] == 
                behavior_type].groupby(['user_id', 'item_id', 'days']).size()
    user_all_btype = pd.DataFrame(user_all_btype, columns=[name])
    user_all_x1 = pd.merge(user_all_xy, user_all_btype, left_on=
                ['user_id', 'item_id', 'days'], right_index=True, how='left')
    user_all_x1[name].fillna(0, inplace=True)
    return user_all_x1, columns

# f2：构建特征2——
# f3：构建特征3——
# f4: 构建特征4——用户全时间购买量（将来添加rolling
def f4(user_all_xy, columns=[]):
    name = 'x4'#特征名
    columns.append(name) #加入特征名
    user_bought_size_pd=pd.DataFrame(user_bought_size, columns=['x4'])
    user_all_x4 = pd.merge(user_all_xy, user_bought_size_pd, left_on=
                           ['user_id'], right_index=True, how='left')
    user_all_x4['x4'].fillna(0, inplace=True)
    return user_all_x4, columns

# f5: 
# f6: 构建特征6——item_id全时间销量（将来添加rolling
def f6(user_all_xy, columns=[]):
    name = 'x6'#特征名
    columns.append(name) #加入特征名
    item_bought_size_pd = pd.DataFrame(item_bought_size, columns=['x6'])
    user_all_x6 = pd.merge(user_all_xy, item_bought_size_pd, left_on=['item_id'], right_index=True, how='left')
    user_all_x6['x6'].fillna(0, inplace=True)
    return user_all_x6, columns

# f7: 
# f8: 
# f9: 

# f10: 特征10——是否出现在深夜1~8点
def f10(user_all_xy, columns=[]):
    name = 'x10'#特征名
    columns.append(name) #加入特征名
    user_all_x10 = user_all_xy
    user_all_x10.loc[(user_all_x10['hours'] > 0) & (user_all_x10['hours'] < 9), 'x10'] = 0
    user_all_x10['x10'].fillna(1, inplace=True)
    return user_all_x10, columns

# f11_2:

