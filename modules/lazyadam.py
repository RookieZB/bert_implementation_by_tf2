# -*- coding: utf-8 -*-

"""
LazyAdam Optimizer with Weight Decay
https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lazy_adam.py

"""

import tensorflow as tf
from modules import adamw


class LazyAdam(adamw.AdamW):
    def __init__(self, **kwargs):
        super(LazyAdam, self).__init__(**kwargs)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None, **kwargs):
        devi1, type1 = var.device, var.dtype.base_dtype
        coef1 = ((apply_state or {}).get((devi1, type1)) or self._fallback_apply_state(devi1, type1))
        m1, v1 = self.get_slot(var, 'm'), self.get_slot(var, 'v')
        msli1 = coef1['beta_1_t']*tf.gather(m1, indices)+coef1['one_minus_beta_1_t']*grad
        vsli1 = coef1['beta_2_t']*tf.gather(v1, indices)+coef1['one_minus_beta_2_t']*tf.square(grad)
        m2 = self._resource_scatter_update(m1, indices, msli1)
        v2 = self._resource_scatter_update(v1, indices, vsli1)

        with tf.control_dependencies([m2, v2]):
            u1 = -coef1['lr_t']*msli1/(tf.sqrt(vsli1)+coef1['epsilon'])
            u2 = u1-coef1['lr_t']*tf.gather(var, indices)*self.drate
            return tf.group(*[self._resource_scatter_add(var, indices, u2), m2, v2])
