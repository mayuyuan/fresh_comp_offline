from offlineclass import *
# 读取并处理经过特征工程获得的train和test数据
item_id_all = read_data('item_all')[['item_id']].drop_duplicates()
trainxy_all = pd.read_csv('data/trainxy_all.csv', dtype={'user_id': str, 'item_id': str})
testxy_all = pd.read_csv('data/testxy_all.csv', dtype={'user_id': str, 'item_id': str})
# 调整分类结果正负比例，↓使其1:1
train_y1 = trainxy_all[trainxy_all['y'] == 1]
train_y0 = trainxy_all[trainxy_all['y'] != 1]
#train_y0['y']=-1 #采用tanh的激励函数，所以二分类用±1
trainxy_all = pd.concat([train_y1, train_y0.sample(train_y1.shape[0])])
trainxy_all = trainxy_all.sample(frac = 1) #乱序

f_col=list(trainxy_all.columns)
f_col.remove('y')
test_X = testxy_all[f_col].values
#%%
#直接tensorflow
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32, [None, len(f_col)], name='x_input')
    ys=tf.placeholder(tf.float32, [None,], name='y_input')
with tf.name_scope('middlelayer'):
    n_layer = 40
    output_size = [64]*2+[32]*(n_layer-2)
    keep_prob = tf.placeholder(tf.float32, [n_layer,], name='keep_prob')
    activation_function = [tf.nn.relu]*n_layer
    middlelayer = multilayer(n_layer, xs, output_size=output_size, 
                    keep_prob=keep_prob, activation_function=activation_function)
y_pre =layer('layer_y', middlelayer, output_size=1, activation_function=tf.nn.relu)
# loss
with tf.name_scope('loss'):
    cross_entropy = -tf.reduce_mean(ys * tf.log(tf.clip_by_value(y_pre,1e-10,1.0)))
#    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_pre-ys), reduction_indices=[1]))
    loss = cross_entropy
    tf.summary.scalar('cross_entropy', loss)

# optimizer
with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(0.0001).minimize(loss)

#evaluate
with tf.name_scope('evaluate'):
    with tf.name_scope('correct_prediction'):
#        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(ys, 1))
        correct_prediction=tf.equal(ys, y_pre)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('value', accuracy)
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
#%%
# 记得close sess，要不然很多东西会保留下来影响你二次运行程序
sess = tf.Session()
sess.run(init)
writer=tf.summary.FileWriter("./log", sess.graph)
# 运行后，会在相应的目录里生成一个文件，执行：tensorboard --logdir='./log'
#%%
##########开始训练##########
#模型训练好多好多周期,用minibatch，一次数万个确实太大了
train_xs=trainxy_all[f_col].values
train_ys=trainxy_all['y'].values
batch = 1000
n = 0
len_train = len(train_xs)
for i in range(3000):
#feed的是numpy.ndarray格式
    if n+batch < len_train:
        batch_xs = train_xs[n:n+batch]
        batch_ys = train_ys[n:n+batch]
        n = n+batch
    else:
        batch_xs = np.vstack((train_xs[n:], train_xs[:n+batch-len_train]))
        batch_ys = np.hstack((train_ys[n:], train_ys[:n+batch-len_train])) #hstack
        n = n+batch-len_train
    sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys, keep_prob:[0.6]*n_layer})
    if i%200 == 0:
        results = sess.run(merged, feed_dict={xs:batch_xs, ys: batch_ys, keep_prob:[1]*n_layer})
        writer.add_summary(results, i)
#%%
# 测试第30天
testxy_all['y'] = sess.run(y_pre, feed_dict={xs:test_X, keep_prob:[1]*n_layer})
result = testxy_all.sort_values(by='y', ascending=False)
result = result[['user_id', 'item_id']].head(20000).drop_duplicates()
# 去除非item_all集合内的item_id
item_id_all['bool'] = 1
result = pd.merge(result, item_id_all, left_on=['item_id'], right_on=['item_id'], how='inner').drop('bool', axis=1)
#%%
########################## 存储答案 ##########################
result.to_csv('results/answer'+datetime.now().strftime('%Y-%m-%d')+'.csv', encoding='utf-8', index=False)
result.to_csv('results/tianchi_mobile_recommendation_predict.csv', encoding='utf-8', index=False)
