import tensorflow as tf
import numpy as np

def mse(pred, true) :
    assert pred.shape[0] == true.shape[0], 'Length of predicted value and GT are not same'
    diff = tf.subtract(pred, true)
    power = tf.square(diff)
    result = tf.math.reduce_mean(power)
    return result

def rmse(pred, true) :
    assert pred.shape[0] == true.shape[0], 'Length of predicted value and GT are not same'
    
    pred_v = pred[:,0]
    pred_a = pred[:,1]

    true_v = true[:,0]
    true_a = true[:,1]
    
    diff_v = tf.subtract(pred_v, true_v)
    diff_a = tf.subtract(pred_a, true_a)

    power_v = tf.square(diff_v)
    power_a = tf.square(diff_a)

    summ_v = tf.math.reduce_mean(power_v)
    summ_a = tf.math.reduce_mean(power_a)

    result_v = tf.sqrt(summ_v)
    result_a = tf.sqrt(summ_a)

    return result_v, result_a


if __name__ == '__main__' :
    pred = np.array([[3.0,2.0], [5.0,0.0], [4.0, 7.0]])
    true = np.array([[1.0,1.0], [2.0,2.0], [3.0, 3.0]])

    pred_1 = np.array([[4, 7]])
    true_1 = np.array([[3, 3]])
    print(rmse(pred,true)[0].numpy())

    ''' 
    a = (2.0, 1.0)
    b = (3.0, 2.0)
    c = (1.0, 4.0)

    print(np.sqrt((a[0]**2 + b[0]**2 + c[0]**2)/3))
    print(np.sqrt((a[1]**2 + b[1]**2 + c[1]**2)/3))
    
    d = [1, 2, 3, 4, 5]
    print(np.power(d, 2).sum()/len(d))
    
    e = []
    e.append(a[0])
    e.append(b[0])
    e.append(c[0])
    print(sqrt(power(e, 2).sum()/len(e)))
    #print(sum(e)/len(e))
    '''
    #print(mse(pred, true))
