import tensorflow as tf

def ffGE(t):
    piece = t/.710649
    shape = tf.square(1. + piece)
    GE = 1./shape
    return GE

def ffGM(t):
    shape = ffGE(t)
    GM0 = 2.792847337
    GM =  GM0*shape
    return GM

def ffF2(t):
    f2 = (ffGM(t) - ffGE(t))/(1-(t/(4*.938*.938)))
    return f2

def ffF1(t):
    f1 = ffGM(t) - ffF2(t)
    return f1

def ffGA(t):
    ga = 1.2695
    ma = 1.026
    part = t/(ma*ma)
    dif = tf.square(1. - part)
    GA = ga/dif
    return GA