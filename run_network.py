import os
import time
import math
import argparse
import termcolor
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()
import BHDVCS_tf


import numpy as np

import network_utils as utils
import theory_bound, net, graphtrace, xsx
#import BHDVCS
from data import data_utils as data


config = utils.parse_config(utils.package_relative_path('config.ini'))
learning_rate = config['TRAINING']['learning_rate']
batch_size = config['TRAINING']['batch_size']
opt_type = config['TRAINING']['optimizer']
save_freq = config['TRAINING']['save_freq']
weight_norm_penalty = config['NETWORKS']['weight_norm_penalty']
dropout_fraction = config['NETWORKS']['dropout_fraction']
hidden_arch_list = config['NETWORKS']['network_hidden']
clip_norm = config['TRAINING']['clip_norm']

bhdvcs = BHDVCS_tf.BHDVCS()


def train(epochs, run_name, verbosity=2, validation_set=True, line_num = -1):
    # reset trace record. will give warning if a function is being traced too often
    graphtrace.reset_trace_record()

    log_dir, checkpoint_dir = utils.make_results_dirs(run_name)

    # create tensorflow model
    model = net.make_model(hidden_arch_list, weight_norm_penalty, dropout_fraction)

    if verbosity > 1: model.summary()
    # save basic diagram of model architecture for reference
    #if verbosity > 1:
        #tf.keras.utils.plot_model(model, to_file=os.path.join(log_dir, 'model.png'), show_shapes=True)

    if opt_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_type == 'rms':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif opt_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Optimizer type '{opt_type}' not recognized.")

    # load dataset
    train_set, val_set = data.full_data_load(line_num=line_num)
    train_loss_list = []
    val_loss_list = []

    num_train = int(len(train_set)/batch_size)
    num_val = int(len(val_set)/batch_size)
    #train_set = train_set[np.where((train_set[:,-3] == 2) | (train_set[:,-3] == 2) | (train_set[:,-3] == 2))]
    #val_set = val_set[np.where((val_set[:,-3] == 2) | (val_set[:,-3] == 2) | (val_set[:,-3] == 2))]

    # dataset format is experiment_id, x, t, Q2, k0, phi, L, sigma, error
    # for now we are not using the experiment_id value
    kin_train, sig_train = train_set[:,1:5], train_set[:,5:14]
    kin_val, sig_val = val_set[:,1:5], val_set[:,5:14]

    if not validation_set:
        kin_train = np.concatenate([kin_train, kin_val], axis=0)
        sig_train = np.concatenate([sig_train, sig_val], axis=0)

    # need batch_count for progress bar
    batch_count = math.ceil(data.get_sample_count(kin_train)/batch_size)
    # convert numpy arrays to tf.data.Datasets
    train_dataset = data.create_dataset(kin_train, sig_train, batch_size)
    if validation_set: val_dataset = data.create_dataset(kin_val, sig_val, batch_size)

    # tensorboard file writer
    if verbosity > 0: log_writer = tf.contrib.summary.create_file_writer(log_dir)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    grad_norm = tf.keras.metrics.Mean('grad_norm', dtype=tf.float32)
    trace_graph = graphtrace.TRACE_GRAPHS

    # main training loop
    for epoch in range(epochs):
        start_time = time.time()
        if verbosity > 1: progbar = tf.keras.utils.Progbar(batch_count)
        for (batch, (kinematics, sigma_true)) in enumerate(train_dataset):
            #print(batch)
            if batch == batch_count:
                break
            #print(kinematics)
            with tf.GradientTape() as tape:
                # neural network maps kinematics --> form factors
                #print(kinematics.shape)
                ffs = predict(model, kinematics, training=True)
                if trace_graph: tf.summary.trace_on(graph=True, profiler=True)
                #print('Form Factors: ', ffs)
                # calculate loss based on most recent predictions
                loss = loss_function(kinematics, ffs, bhdvcs, sigma_true)
                # add in regularization loss
                loss += sum(model.losses)
                #print('Loss ', loss)
                if trace_graph and verbosity > 0:
                    with log_writer.as_default():
                        tf.summary.trace_export(
                            name="Inner Loop Trace",
                            step=0,
                            profiler_outdir=log_dir
                        )
                        trace_graph = False

            # get gradients of loss w.r.t trainable variables
            gradients = tape.gradient(loss, model.trainable_variables)
            #print(gradients)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, clip_norm)
            grad_norm(norm)
            if verbosity > 0 and False:
                with log_writer.as_default():
                # tensorboard heatmaps, histograms
                    current_step = (epoch*batch_count)+batch
                    if verbosity > 2:
                        ffs_img = utils.figure_to_image(utils.form_factor_figure(ffs))
                        tf.summary.image('Form Factor Predictions', ffs_img)#, step=current_step)
                    tf.compat.v1.summary.histogram("loss", loss)#, step=current_step)
                    for num, grad in enumerate(gradients):
                        tf.summary.histogram(f"grad_{num}", grad)#, step=current_step)
            # update trainable variables based on calculated gradients
            optimizer.apply_gradients(zip(clipped_gradients, model.trainable_variables))
            train_loss(loss)
            if verbosity > 1: progbar.update(batch+1)
        end_time = time.time()
        #print('step 1')
        if validation_set:
            #print(val_dataset)
            val_count = 0
            for kinematics_v, sigma_true_v in val_dataset:
                # calculate validation loss
                if val_count >= num_val:
                    break
                val_count+=1
                ffs_v = predict(model, kinematics_v, training=False)
                loss_v = loss_function(kinematics_v, ffs_v, bhdvcs, sigma_true_v)
                loss_v += sum(model.losses)
                val_loss(loss_v)
        #print('step 2')
        if verbosity > 0:
            #with log_writer.as_default():
                # tensorboard scalars
            tf.summary.scalar('train_loss', train_loss.result())#, step=epoch)
            if validation_set: tf.summary.scalar('val_loss', val_loss.result())#, step=epoch)
            tf.summary.scalar('grad_norm', grad_norm.result())#, step=epoch)
        #print('step 3')
        if verbosity >= 1:
            # one-line summary of this epoch
            t_loss_str = utils.tensor_to_colored_string(train_loss, 'green')
            v_loss_str = utils.tensor_to_colored_string(val_loss, 'red') if validation_set else "N/A"
            train_loss_list.append(train_loss.result().numpy())
            val_loss_list.append(val_loss.result().numpy())
            print(f"Epoch {epoch+1}, Train_Loss {t_loss_str}, Val_Loss {v_loss_str}, Time {end_time-start_time}")
        #print('step 4')
        if epoch < epochs-1:
            #not on the final iteration
            train_loss.reset_states()
            val_loss.reset_states()
            grad_norm.reset_states()
        #print('step 5')
        # save checkpoints every save_freq epochs and on the final epoch
        if (epoch % save_freq == 0 and epoch > 0) or epoch == epochs-1:
            save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}", 'weights')
            model.save_weights(save_path)
    
    ## quick test:
    sum_reh = 0.0
    sum_ree = 0.0
    sum_reht = 0.0
    first = True
    num=0
    for (batch, (kinematics, sigma_true)) in enumerate(train_dataset):
        ffs = predict(model, kinematics, training=True)
        reH, reE, reHt = tf.unstack(ffs, axis=-1)

        reH, reE, reHt = theory_bound.theory_bound(reH, reE, reHt)
        k0, Q2, xbj, t = tf.unstack(kinematics, axis=-1)
        k0 = tf.dtypes.cast(k0, tf.float32)
        Q2 = tf.dtypes.cast(Q2, tf.float32)
        xbj = tf.dtypes.cast(xbj, tf.float32)
        t = tf.dtypes.cast(t, tf.float32)
        phi, L, sigma_true, error, F1, F2, reH_real, reE_real, reHT_real = tf.unstack(sigma_true, axis=-1)
        phi = tf.dtypes.cast(phi, tf.float32)
        sigma_true = tf.dtypes.cast(sigma_true, tf.float32)
        F1 = tf.dtypes.cast(F1, tf.float32)
        F2 = tf.dtypes.cast(F2, tf.float32)
        reH_real = tf.dtypes.cast(reH_real, tf.float32)
        reE_real = tf.dtypes.cast(reE_real, tf.float32)
        reHT_real = tf.dtypes.cast(reHT_real, tf.float32)
        #F2 = tf.dtypes.cast(F2, tf.float32)
        L = tf.dtypes.cast(L, tf.int32)
        sigma_pred = bhdvcs.TotalUUXS([phi], [Q2, xbj, t, k0, F1, F2, reH, reE, reHt, tf.constant(0.014863)])
        sigma_pred = tf.dtypes.cast(sigma_pred, tf.float32)
        sigma_equation = bhdvcs.TotalUUXS([phi], [Q2, xbj, t, k0, F1, F2, reH_real, reE_real, reHT_real, tf.constant(0.014863)])
        sigma_equation = tf.dtypes.cast(sigma_equation, tf.float32)
        error = tf.dtypes.cast(error, tf.float32)
        #sigma_pred = xsx.ff_to_xsx(reH, imH, reE, imE, reHt, imHt, reEt, imEt, phi, xbj, t, Q2, L, k0)
        num+=1
        if first:
            sum_reh = reH
            sum_ree = reE
            sum_reht = reHt
            first = False
        else:
            sum_reh+=reH
            sum_ree += reE
            sum_reht += reHt
            
        if verbosity>2:
            print(sigma_true)
            print('------------------')
            print('Eq with params predictions: ', sigma_pred)
            print('Eq. With Correct Params: ', sigma_equation)
            print('++++++++++++++++++')
            print('Param estimates: ', ffs)
            print('k: ', k0, ' Q2: ', Q2, ' xbj: ', xbj, ' t: ', t)
            print('==================')
            
            print('~~~~~~~~~~~~~~~~~~')

    ### loss graph
    #plt.plot(np.arange(len(train_loss_list)), train_loss_list, color='r-', label='Train')
    #plt.plot(np.arange(len(val_loss_list)), val_loss_list, color='b-', label='Validation')
    #plt.legend()
    #plt.show()
    print('True reH: ', reH_real)
    print('True reE: ', reE_real)
    print('True reHT: ', reHT_real)
    print('Avg reH: ', sum_reh/num)
    print('Avg reE: ', sum_ree/num)
    print('Avg reHT: ', sum_reht/num)
    print('~~~~~~~~~~~~~~~~~~')


    base_dir = os.path.basename(os.path.dirname(log_dir))
    return val_loss.result().numpy(), base_dir

@graphtrace.trace_graph
def predict(model, kinematics, training):
    return model(kinematics, training)

@graphtrace.trace_graph
def loss_function(kinematics, ffs, bhdvcs_func, sigma_true):
    """Calculate loss for one prediction batch.

    Args:
        kinematics (tf.Tensor) : Tensor of shape (batch_size, 4)
            consisting of xbj, t, Q2 and phi in that order.
        ffs (tf.Tensor) : Tensor of shape (batch_size, 8) consisting
            of form factor predictions outputted by the model.
        sigma_true (tf.Tensor) : Tensor of shape (batch_size, 3)
            consisting of L, sigma_true, error.

    Returns:
        loss (tf.Tensor) the error-adjusted mean squared error of the
        model's predictions.
    """
    reH, reE, reHt = tf.unstack(ffs, axis=-1)

    reH, reE, reHt = theory_bound.theory_bound(reH, reE, reHt)
    k0, Q2, xbj, t = tf.unstack(kinematics, axis=-1)
    k0 = tf.dtypes.cast(k0, tf.float32)
    Q2 = tf.dtypes.cast(Q2, tf.float32)
    xbj = tf.dtypes.cast(xbj, tf.float32)
    t = tf.dtypes.cast(t, tf.float32)
    phi, L, sigma_true, error, F1, F2, reH_real, reE_real, reHT_real = tf.unstack(sigma_true, axis=-1)
    phi = tf.dtypes.cast(phi, tf.float32)
    sigma_true = tf.dtypes.cast(sigma_true, tf.float32)
    F1 = tf.dtypes.cast(F1, tf.float32)
    F2 = tf.dtypes.cast(F2, tf.float32)
    reH_real = tf.dtypes.cast(reH_real, tf.float32)
    reE_real = tf.dtypes.cast(reE_real, tf.float32)
    reHT_real = tf.dtypes.cast(reHT_real, tf.float32)
    L = tf.dtypes.cast(L, tf.int32)
    sigma_pred = bhdvcs_func.TotalUUXS([phi], [Q2, xbj, t, k0, F1, F2, reH, reE, reHt, tf.constant(0.014863)])
    sigma_pred = tf.dtypes.cast(sigma_pred, tf.float32)
    sigma_equation = bhdvcs.TotalUUXS([phi], [Q2, xbj, t, k0, F1, F2, reH_real, reE_real, reHT_real, tf.constant(0.014863)])
    sigma_equation = tf.dtypes.cast(sigma_equation, tf.float32)
    error = tf.dtypes.cast(error, tf.float32)
    return .1 * tf.math.reduce_mean(
        tf.math.square( (sigma_equation - sigma_pred) / (1. + error) )
    )



if __name__ == '__main__':
    num_epochs = int(input('Please enter number of epochs: '))
    run_name = input('Please enter name of run: ')
    line_number = int(input('Please enter which set of parameters to use\n(Numbers based on which set of 36 points it is in the csv file, starting with 0.\n The line number 1 does not work for some reason, unclear why this is as the data is the same format)'))
    
    #num_epochs = int(input('Please enter number of epochs: '))
    
    train(num_epochs, run_name, line_num=line_number)