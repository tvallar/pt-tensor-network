
import tensorflow as tf



def _bound(val, low, high):
    return tf.math.minimum(tf.math.maximum(val, low),high)

#TODO: get exact constraints from theory group
rangeREH = (0.1, 10)
rangeIMH = (0.1, 10)
rangeREE = (0.1, 10)
rangeIME = (0.1, 10)
rangeREHT = (0.1, 10)
rangeIMHT = (0.1, 10)
rangeREET = (0.1, 10)
rangeIMET = (0.1, 10)

def theory_bound(reH, reE, reHt):
    reH = _bound(reH, *rangeREH)
    #imH = _bound(imH, *rangeIMH)
    reE = _bound(reE, *rangeREE)
    #imE = _bound(imE, *rangeIME)
    reHt = _bound(reHt, *rangeREHT)
   # imHt = _bound(imHt, *rangeIMHT)
    #reEt = _bound(imEt, *rangeREET)
    #imEt = _bound(reEt, *rangeIMET)
    return reH, reE, reHt
