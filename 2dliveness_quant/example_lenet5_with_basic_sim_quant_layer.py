#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import sys
import os
import importlib
from tensorflow.python.framework import ops

ndk_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(ndk_dir)

import ndk
from ndk.simulated_quant. sim_quant_layers import  SimQuantLayerInput,SimQuantLayerDense,SimQuantLayerConv2d, \
    SimQuantLayerRelu, SimQuantLayerPool, SimQuantLayerBatchNorm
from ndk.simulated_quant.sim_quant_model import SimQuantModel
from ndk.utils import print_log
import ndk.examples.data_generator_mnist as mnist
import ndk.examples.data_generator_imagenet_partial as ln

def construct_lenet5(net_input):
    model = SimQuantModel()

    sim_quant_layer = SimQuantLayerInput(input=(net_input, net_input.name),
                                         name='net_input',
                                         bitwidth=16,
                                         dim=(1, 1, 112, 96))

    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerConv2d(input=sim_quant_layer.output,
                                          name='conv1',
                                          bitwidth=16,
                                          trainable=True,
                                          num_output=16,
                                          kernel_size=3,
                                          stride=2,
                                          pad=(0,1,0,1))
    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerBatchNorm(input=sim_quant_layer.output,
                                             bitwidth=16,
                                             name='bn1')

    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerRelu(input=sim_quant_layer.output,
                                        name='relu1',
                                        bitwidth=16)

    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerConv2d(input=sim_quant_layer.output,
                                          name='conv2',
                                          bitwidth=16,
                                          trainable=True,
                                          num_output=16,
                                          kernel_size=3,
                                          stride=2,
                                          pad=(0,1,0,1))
    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerBatchNorm(input=sim_quant_layer.output,
                                             bitwidth=16,
                                             name='bn2')
    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerRelu(input=sim_quant_layer.output,
                                        name='relu2',
                                        bitwidth=16)

    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerConv2d(input=sim_quant_layer.output,
                                          name='conv3',
                                          bitwidth=16,
                                          trainable=True,
                                          num_output=32,
                                          kernel_size=3,
                                          stride=2,
                                          pad=(0,1,0,1))
    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerRelu(input=sim_quant_layer.output,
                                        name='relu3',
                                        bitwidth=16)
    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerBatchNorm(input=sim_quant_layer.output,
                                             bitwidth=16,
                                             name='bn3')

    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerConv2d(input=sim_quant_layer.output,
                                          name='conv4',
                                          bitwidth=16,
                                          trainable=True,
                                          num_output=32,
                                          kernel_size=3,
                                          stride=2,
                                          pad=(0,1,0,1))
    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerBatchNorm(input=sim_quant_layer.output,
                                             bitwidth=16,
                                             name='bn4')
    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerRelu(input=sim_quant_layer.output,
                                        name='relu4')


    model.add_layer(sim_quant_layer)
    sim_quant_layer = SimQuantLayerDense(input=sim_quant_layer.output,
                                         name='out',
                                         bitwidth=16,
                                         trainable=True,
                                         num_output=2)


    model.add_layer(sim_quant_layer)

    logits = sim_quant_layer.output_tensor

    return logits, model






if __name__ == '__main__':

    FLOAT_MODEL_TRAIN_STEP = 20000
    TRAIN_FLOAT_MODEL = False
    QUANT_GENERAL = True
    QUANT_TRAIN = False
    EVAL_FLOAT = True
    EVAL_QUANT_TRAIN = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    tf.reset_default_graph()
    lr = 0.001

    g_train = ln.data_generator_imagenet_partial(
        imagenet_dirname=r'./liveness/',
        batch_size=512,
        random_order=True,
        n=75776,
        filenames_to_class="train_images.json",
        grayscale=True,
        one_hot=False,
        num_class=2
    )


    if TRAIN_FLOAT_MODEL:

        input_pl = tf.placeholder(dtype=tf.float32,
                                  shape=[None, 1, 112, 96])
        output_pl = tf.placeholder(dtype=tf.float32,
                                   shape=[None, 2])
        logits, model= construct_lenet5(input_pl)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=output_pl,
                                               logits=logits)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss)
        train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(output_pl, 1)), tf.float32))


        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()


            for idx in range(FLOAT_MODEL_TRAIN_STEP):
                if idx % 5000 == 0 and idx != 0 and idx < 10001:
                    lr = lr / 10
                data_batch = next(g_train)

                _, train_loss, acc = sess.run([train_op, loss, train_acc],
                                         feed_dict={
                                         input_pl:data_batch['input'],
                                         output_pl:np.reshape(
                                             data_batch['output'],
                                             newshape=(-1,2))}
                )

                if idx%100 == 0 and idx!=0:
                    print_log('train_acc:{:.8f}, loss:{:.8f}, '
                              'step {}/{}.'.format(
                                  acc, train_loss,  idx, FLOAT_MODEL_TRAIN_STEP))



            layer_list, param_dict =\
                                 model.export_quant_param_dict(sess)


        ndk.modelpack.save_to_file(layer_list,
                                   '2d_model_0615_without',
                                   param_dict,
                                   '2d_model_0615_without')

    if QUANT_GENERAL:
        layer_list, param_dict = ndk.tensorflow_interface.load_from_pb("spc_2d_0713_v2.pb")

        layer_list, param_dict = ndk.optimize.merge_layers(layer_list, param_dict)

        quant_layer_list, quant_param_dict = ndk.quantize.quantize_model(layer_list=layer_list,
                                                                         param_dict=param_dict,
                                                                         bitwidth=16,
                                                                         data_generator=g_train,
                                                                         usr_param_dict=None,
                                                                         num_step_pre=20,
                                                                         num_step=120,
                                                                         gaussian_approx=False
        )

        fname = 'spc_2d_0713_quant_v2'
        ndk.modelpack.save_to_file(layer_list=quant_layer_list, fname_prototxt=fname,
                                   param_dict=quant_param_dict, fname_npz=fname)

    if QUANT_TRAIN:
        fname = 'spc_2d_0713_quant_v2'
        quant_layer_list, quant_param_dict = ndk.modelpack.load_from_file(fname_prototxt=fname, fname_npz=fname)
        quant_layer_list, quant_param_dict = ndk.quantize.quantize_model_with_training(layer_list=quant_layer_list,
                                                                                       bitwidth=16,
                                                                                       param_dict=quant_param_dict,
                                                                                       data_generator=g_train,
                                                                                       loss_fn=tf.losses.softmax_cross_entropy,
                                                                                       optimizer=tf.train.AdamOptimizer(1e-5),
                                                                                       num_step_train=200,
                                                                                       num_step_log=100,
                                                                                       layer_group=1)
        fname = 'spc_2d_0713_quant_train_v2'
        ndk.modelpack.save_to_file(layer_list=quant_layer_list, fname_prototxt=fname,
                                   param_dict=quant_param_dict, fname_npz=fname)

    if EVAL_FLOAT:
        layer_list, param_dict = ndk.tensorflow_interface.load_from_pb("spc_2d_0713.pb")
        g_test = ln.data_generator_imagenet_partial(
                                imagenet_dirname=r'./liveness/',
                                batch_size=3100,
                                random_order=False,
                                n=3100,
                                filenames_to_class="test_images.json",
                                interpolation='bilinear',
                                grayscale=True,
                                one_hot=False,
                                num_class=2
                            )

        data_batch = next(g_test)
        net_output_tensor_name = ndk.layers.get_network_output(layer_list)[0]
        test_output = ndk.quant_tools.numpy_net.run_layers(input_data_batch=data_batch['input'],
                                                           layer_list=layer_list,
                                                           target_feature_tensor_list=[net_output_tensor_name],
                                                           param_dict=param_dict,
                                                           bitwidth=16,
                                                           quant=False,
                                                           hw_aligned=True,
                                                           numpy_layers=None,
                                                           log_on=True)
        correct_prediction = np.equal(np.argmax(test_output[net_output_tensor_name], 1), np.argmax(data_batch['output'], 1))
        accuracy = np.mean(correct_prediction)
        print('Before quantization, model accuracy={:.3f}%'.format(accuracy*100))

    if EVAL_QUANT_TRAIN:

        fname = 'spc_2d_0713_quant_v2'
        quant_layer_list, quant_param_dict = ndk.modelpack.load_from_file(fname_prototxt=fname, fname_npz=fname)

        g_test = ln.data_generator_imagenet_partial(
            imagenet_dirname=r'./liveness/',
            batch_size=3100,
            random_order=False,
            n=3100,
            filenames_to_class="test_images.json",
            interpolation='bilinear',
            grayscale=True,
            one_hot=False,
            num_class=2
                            )

        data_batch = next(g_test)
        net_output_tensor_name = ndk.layers.get_network_output(quant_layer_list)[0]
        test_output = ndk.quant_tools.numpy_net.run_layers(input_data_batch=data_batch['input'],
                                                           layer_list=quant_layer_list,
                                                           target_feature_tensor_list=[net_output_tensor_name],
                                                           param_dict=quant_param_dict,
                                                           bitwidth=16,
                                                           quant=True,
                                                           hw_aligned=True,
                                                           numpy_layers=None,
                                                           log_on=True)

        infer = np.argmax(test_output[net_output_tensor_name], 1)
        labels = np.argmax(data_batch['output'], 1)

        for i in range(3100):
            if infer[i] != labels[i]:
                print(data_batch['filenames'][i])

        correct_prediction = np.equal(np.argmax(test_output[net_output_tensor_name], 1), np.argmax(data_batch['output'], 1))
        accuracy = np.mean(correct_prediction)
        print('After quantization, model accuracy={:.3f}%'.format(accuracy*100))
