#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LISTA_base.py
author: xhchrn
email : chernxh@tamu.edu
date  : 2018-10-03

A base class for all LISTA networks.
"""

import numpy as np
import numpy.linalg as la
import tensorflow as tf

import utils.train

class LISTA_base (object):

    """
    Implementation of deep neural network model.
    """

    def __init__ (self):
        pass

    def setup_layers (self):
        pass

    def inference (self):
        pass

    def save_trainable_variables (self , sess , savefn):
        """
        Save trainable variables in the model to npz file with current value of each
        variable in tf.trainable_variables().

        :sess: Tensorflow session.
        :savefn: File name of saved file.

        """
        state = getattr (self , 'state' , {})
        utils.train.save_trainable_variables (
                sess, savefn, self.scope_name, **state )

    def load_trainable_variables (self, sess, savefn):
        """
        Load trainable variables from saved file.

        :sess: TODO
        :savefn: TODO
        :returns: TODO

        """
        self.state = utils.train.load_trainable_variables (sess, savefn)

    def do_training (self, sess, stages, savefn,
                     batch_size, val_step, maxit, better_wait):
        """
        Do training actually. Refer to utils/train.py.

        :sess       : Tensorflow session, in which we will run the training.
        :stages     : List of tuples. Training stages obtained via
            `utils.train.setup_training`.
        :savefn     : String. Path where the trained model is saved.
        :batch_size : Integer. Training batch size.
        :val_step   : Integer. How many steps between two validation.
        :maxit      : Integer. Max number of iterations in each training stage.
        :better_wait: Integer. Jump to next stage if no better performance after
            certain # of iterations.

        """
        self.state = utils.train.do_training (
                sess, stages, savefn, self._scope, val_step, maxit, better_wait)


    # def do_cs_training (self, sess,
    #                     train_y_, train_f_, train_x_,
    #                     val_y_  , val_f_  , val_x_,
    #                     savefn, batch_size=64, val_step=10,
    #                     maxit=200000, better_wait=4000, norm_patch=False) :
    #     """
    #     Do training on compressive sensing problem actually. Refer to
    #     utils/train.py.

    #     Param:
    #         :sess    : Tensorflow session.
    #         :trainfn : Path of training data tfrecords.
    #         :valfn   : Path of validation data tfrecords.
    #         :savefn  : Path that trained model to be saved.
    #     Hyperparam:
    #         :batch_size : Batch size.
    #         :val_step   : How many steps between two validation.
    #         :maxit      : Max number of iterations in each training stage.
    #         :better_wait: Jump to next stage if no better performance after
    #                       certain # of iterations.
    #     """
    #     self.state = utils.train.do_cs_training (
    #             sess, self.stages, self.prob,
    #             train_y_, train_f_, train_x_,
    #             val_y_  , val_f_  , val_x_,
    #             savefn, self.scope_name,
    #             batch_size, val_step, maxit, better_wait, norm_patch)


