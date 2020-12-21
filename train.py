import tensorflow as tf
from src.ResNet import ResNet34
from src.basic import VGG, AlexNet
from metrics import mse, rmse

import h5py
import os
import pandas as pd

'''
################### Limit GPU Memory ###################
gpus = tf.config.experimental.list_physical_devices('GPU')
print("########################################")
print('{} GPU(s) is(are) available'.format(len(gpus)))
print("########################################")

# set the only one GPU and memort limit
memory_limit = 1024*5

if gpus :
    try :
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = memory_limit)])
        print("Use only one GPU{} limited {}MB memory".format(gpus[0], memory_limit))
    except RuntimeError as e :
        print(e)

else :
    print('GPU is not available')
##########################################################
'''

@tf.function
def train_step(ref_model, X, Y) :
    with tf.GradientTape() as tape :
        predictions = ref_model(X)
        loss = LOSS(predictions, Y)
        metric = METRIC(predictions, Y)

    gradients = tape.gradient(loss, ref_model.trainable_variables)
    OPTIMIZER.apply_gradients(zip(gradients, ref_model.trainable_variables))

    return ref_model, loss, metric

@tf.function
def val_step(ref_model, X, Y) :
    predictions = ref_model(X)
    loss = LOSS(predictions, Y)
    metric = METRIC(predictions, Y)

    return loss, metric

# Model Setting
BASE_MODEL = 'ResNet'
LAYERS = 18
SENet = None  # CA, SA, serial_CA_SA, serial_SA_CA, parallel_mul, parallel_add
ADVENCED = False  # if True, Final FC layer is increaed.

# Hyper-parameter setting
EPOCH = 10
TRAIN_BATCH = 10000
BATCH = 64
OPTIMIZER = tf.keras.optimizers.Adam()
LOSS = mse
METRIC = rmse



# Path setting
pwd = os.path.abspath( __file__ )
base_dir = os.path.dirname(pwd)
data_dir = os.path.join(base_dir, 'AffectNet')
save_dir = os.path.join(base_dir, 'Train4Paper')

# Load Data
train_path = os.path.join(data_dir, 'train_normal.h5')
val_path = os.path.join(data_dir, 'validation_normal.h5')

train = h5py.File(train_path, 'r')
val = h5py.File(val_path, 'r')

x_train = train['image']
y_train = train['VA']

x_val = val['image']
y_val = val['VA']

print('train input data is {}'.format(x_train.shape))
print('train output data is {}'.format(y_train.shape))
print('validation input data is {}'.format(x_val.shape))
print('validation output data is {}'.format(y_val.shape))


# Load Model
print("[INFO] creating model...")
if BASE_MODEL == 'ResNet' :
    cardinality = None
    model = ResNet34(num_layer = LAYERS, cardinality = cardinality, se = SENet, adv = ADVENCED)
elif BASE_MODEL == 'ResNeXt' :
    if LAYERS == 34 : 
        cardinality = 32
        model = ResNet34(cardinality = cardinality, se = SENet, adv = ADVENCED)
elif BASE_MODEL == 'VGG' :
    model = VGG(LAYERS)
elif BASE_MODEL == 'AlexNet' : 
    model = AlexNet()

if not SENet == None :
    if not ADVENCED :
        model_name = BASE_MODEL + str(LAYERS) + '_' + SENet
    else :
        model_name = BASE_MODEL + str(LAYERS) + '_' + SENet + '_' + 'ADV'
else :
    if not ADVENCED : 
        if BASE_MODEL == 'AlexNet' :
            model_name = BASE_MODEL
        else :
            model_name = BASE_MODEL + str(LAYERS)
    else :
        model_name = BASE_MODEL + str(LAYERS) + '_' + 'ADV'

save_path = os.path.join(save_dir, model_name)

if not os.path.isdir(save_path) :
    os.mkdir(save_path)

'''
weight_path = os.path.join(base_dir, 'weights', 'ResNet34', 'checkpoint_8_300000-320739.ckpt')
model.load_weights(weight_path)
'''

#tf.config.threading.set_intra_op_parallelism_threads(10)
#tf.config.threading.set_inter_op_parallelism_threads(10)
print('=========================')
print(tf.config.threading.get_intra_op_parallelism_threads())
print(tf.config.threading.get_inter_op_parallelism_threads())
print('=========================')

# Train
num_train = int(x_train.shape[0] / TRAIN_BATCH)
num_val = int(x_val.shape[0] / BATCH)

result = {'iteration':[],
          'train_loss':[], 
          'val_loss':[],
          'val_valence':[],
          'val_arousal':[]}

for epoch in range(EPOCH) :
    
    # train
    for i in range(num_train+1) :
        print("[INFO] starting epoch {}/{}...{}/{}".format(epoch + 1, EPOCH, i+1, num_train+1))
        start = i * TRAIN_BATCH
        if i == num_train : 
            end = -1
        else :
            end = start + TRAIN_BATCH

        result['iteration'].append(int(epoch*x_train.shape[0] + start))

        X_train = tf.convert_to_tensor(x_train[start:end])
        Y_train = tf.convert_to_tensor(y_train[start:end])
        
        model.compile(OPTIMIZER, loss=LOSS)
        hist = model.fit(X_train, Y_train, epochs=1, batch_size=BATCH)
        
        result['train_loss'].append(hist.history['loss'][0])

        del X_train, Y_train

        # validation
        result_val_loss = []
        result_val_valence = []
        result_val_arousal = []
        for j in range(num_val+1) :
            start_val = j * BATCH
            if j == num_val :
                end_val = -1
            else :
                end_val = start_val + BATCH
    
            X_val = x_val[start_val:end_val]
            Y_val = y_val[start_val:end_val]

            val_loss, val_metric = val_step(model, X_val, Y_val)
        
            result_val_loss.append(val_loss)
            result_val_valence.append(val_metric[0])
            result_val_arousal.append(val_metric[1])

        del X_val, Y_val
        
        temp_loss = tf.math.reduce_mean(result_val_loss).numpy()
        result['val_loss'].append(temp_loss)
        result['val_valence'].append(tf.sqrt(tf.math.reduce_mean(tf.square(result_val_valence))).numpy())
        result['val_arousal'].append(tf.sqrt(tf.math.reduce_mean(tf.square(result_val_arousal))).numpy())

        # print(result)
        print(temp_loss)
        #save weights
        if temp_loss == min(result['val_loss']) : 
            model.save_weights(os.path.join(save_path, "ckpt"))
            print('save weights')

    # save result
    df = pd.DataFrame(result)
    df.to_csv(os.path.join(save_path, 'train_result.csv'), index=False)
