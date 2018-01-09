
########################## 数据准备阶段 ##########################
# 读取商品分类数据，不要地理位置的信息
item_all = read_data('item_all')[['item_id','item_category']].drop_duplicates()
# 读取用户行为数据，不要地理位置的信息
user_all = read_data('user_all').drop('user_geohash', axis=1)
# 所有用户ID的list
user_id_all = user_all['user_id'].drop_duplicates()
#%%
# 买过东西的user,将日期转换为days
user_bought = user_all.loc[user_all['behavior_type'] == '4']
user_bought_size = user_bought.groupby('user_id').size()
item_bought_size = user_bought.groupby('item_id').size()
user_bought['days'] = (user_bought['time'] - datetime(2014, 11, 18)).apply(lambda x: x.days)
#user_bought = user_bought[['user_id', 'item_id', 'days']].drop_duplicates().sort_values(by='days')
# user_id = '64869447'
# item_id = '708284'
#%%
########################## 加入特征，制作训练集和预测集 ##########################
f_col=[] #所有特征名称
user_all_x, user_all = u1(user_all)#运行时间好几min
#当天该pair操作(1,2,3,4)的次数
user_all_x, f_col = f1(user_all_x, behavior_type='1', columns=f_col)
user_all_x, f_col = f1(user_all_x, behavior_type='2', columns=f_col)
user_all_x, f_col = f1(user_all_x, behavior_type='3', columns=f_col)
user_all_x, f_col = f1(user_all_x, behavior_type='4', columns=f_col)
user_all_x, f_col = f4(user_all_x, columns=f_col)
user_all_x, f_col = f6(user_all_x, columns=f_col)
user_all_x, f_col = f10(user_all_x, columns=f_col)
user_all_xy = y1(user_all_x, -1)
#%%
Ds = len(f_col) #特征维数
# 训练集，去除第24天（双十二那天）和最后1天
trainxy_all = user_all_xy.loc[(user_all_xy['days'] <= 29) & (user_all_xy['days'] != 24)]
# 去重？
trainxy_all = trainxy_all[f_col+['y']].drop_duplicates()
# 预测集（第30天）
testxy_all = user_all_xy.loc[user_all_xy['days'] == 30]
# 存储原始train和test数据
trainxy_all.to_csv('data/trainxy_all.csv', index=False)
testxy_all.to_csv('data/testxy_all.csv', index=False)
