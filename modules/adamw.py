# -*- coding: utf-8 -*-

"""
Adam Optimizer with Weight Decay

"""

import tensorflow as tf
import tensorflow.python as ops


class AdamW(tf.keras.optimizers.Adam):
    def __init__(self, step, lrate=1e-3, b1=0.9, b2=0.999, drate=1e-2, name='AdamW', **kwargs):
        super(AdamW, self).__init__(lrate, b1, b2, name=name, **kwargs)
        self.step, self.drate, self.spec = step, drate, ['bias', 'normalization', 'lnorm', 'layernorm']

    @staticmethod
    def _rate_sch(rate, step, total):
        warm1 = total*0.1
        return tf.where(step < warm1, rate*step/warm1, rate*(total-step)/(total-warm1))

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamW, self)._prepare_local(var_device, var_dtype, apply_state)
        rate0 = apply_state[(var_device, var_dtype)]['lr_t']
        rate1 = self._rate_sch(rate0, tf.cast(self.iterations+1, var_dtype), self.step+1)
        apply_state[(var_device, var_dtype)].update(dict(lr=rate1))

    def _resource_apply_base(self, grad, var, indices=None, apply_state=None):
        devi1, type1, name1 = var.device, var.dtype.base_dtype, var.name
        spec1 = any(c1 in name1.lower() for c1 in self.spec)
        coef1 = ((apply_state or {}).get((devi1, type1)) or self._fallback_apply_state(devi1, type1))
        deca1 = tf.no_op if spec1 else var.assign_sub(coef1['lr']*var*self.drate, use_locking=self._use_locking)
        m1, v1 = self.get_slot(var, 'm'), self.get_slot(var, 'v')

        if indices is None:
            with tf.control_dependencies([deca1]):
                return ops.training.training_ops.resource_apply_adam(
                    var.handle, m1.handle, v1.handle, coef1['beta_1_power'], coef1['beta_2_power'], coef1['lr'],
                    coef1['beta_1_t'], coef1['beta_2_t'], coef1['epsilon'], grad, use_locking=self._use_locking)

        m2 = m1.assign(coef1['beta_1_t']*m1, self._use_locking)
        v2 = v1.assign(coef1['beta_2_t']*v1, self._use_locking)

        with tf.control_dependencies([m2, v2, deca1]):
            m2 = self._resource_scatter_add(m1, indices, coef1['one_minus_beta_1_t']*grad)
            v2 = self._resource_scatter_add(v1, indices, coef1['one_minus_beta_2_t']*grad*grad)
            u1 = coef1['lr']*m2/(tf.sqrt(v2)+coef1['epsilon'])
            return tf.group(*[var.assign_sub(u1, self._use_locking), m2, v2])

    def _resource_apply_dense(self, grad, var, apply_state=None, **kwargs):
        return self._resource_apply_base(grad, var, None, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None, **kwargs):
        return self._resource_apply_base(grad, var, indices, apply_state)

    def get_config(self):
        conf1 = super(AdamW, self).get_config()
        conf1.update({'decaying_rate': self.drate, 'step': self.step})
        return conf1
