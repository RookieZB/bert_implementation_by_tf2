# -*- coding: utf-8 -*-

"""
Adam Optimizer with Weight Decay

"""

import tensorflow as tf
import tensorflow.keras as keras


class AdamW(keras.optimizers.Adam):
    def __init__(self, step, lrate=1e-3, drate=1e-2, name='AdamW', sch=True, **kwargs):
        super(AdamW, self).__init__(learning_rate=lrate, name=name, **kwargs)
        self.step, self.drate, self.sch, self.spec = step, drate, sch, ['bias', 'normalization', 'lnorm', 'layernorm']

    @staticmethod
    def _rate_sch(rate, step, total):
        warm1 = total*0.1
        return tf.where(step < warm1, rate*step/warm1, rate*(total-step)/(total-warm1))

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamW, self)._prepare_local(var_device, var_dtype, apply_state)
        rate1 = self._rate_sch(1., tf.cast(self.iterations+1, var_dtype), self.step+1) if self.sch else 1.
        apply_state[(var_device, var_dtype)]['lr_t'] *= rate1
        apply_state[(var_device, var_dtype)]['lr'] *= rate1

    def _resource_apply_base(self, var, apply_state=None):
        devi1, type1, name1 = var.device, var.dtype.base_dtype, var.name
        spec1 = any(c1 in name1.lower() for c1 in self.spec)
        coef1 = ((apply_state or {}).get((devi1, type1)) or self._fallback_apply_state(devi1, type1))
        return tf.no_op if spec1 else var.assign_sub(coef1['lr_t']*var*self.drate, use_locking=self._use_locking)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        deca1 = self._resource_apply_base(var, apply_state)

        with tf.control_dependencies([deca1]):
            return super(AdamW, self)._resource_apply_dense(grad, var, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        deca1 = self._resource_apply_base(var, apply_state)

        with tf.control_dependencies([deca1]):
            return super(AdamW, self)._resource_apply_sparse(grad, var, indices, apply_state)

    def get_config(self):
        conf1 = super(AdamW, self).get_config()
        conf1.update({'decaying_rate': self.drate, 'step': self.step})
        return conf1


class TestAdamW(keras.optimizers.Optimizer):
    def __init__(self, step, lrate=1e-3, b1=0.9, b2=0.999, drate=1e-2, lmode=0, ldecay=None, name='AdamW', **kwargs):
        super(TestAdamW, self).__init__(name, **kwargs)
        self.step, self.drate, self.lmode, self.ldecay, self.sch, self.epsilon = step, drate, lmode, ldecay, True, 1e-6
        self.spec = ['bias', 'normalization', 'lnorm', 'layernorm']
        self._set_hyper('learning_rate', lrate)
        self._set_hyper('beta_1', b1)
        self._set_hyper('beta_2', b2)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')

    @staticmethod
    def _rate_sch(rate, step, total):
        warm1 = total*0.1
        return tf.where(step < warm1, rate*step/warm1, rate*(total-step)/(total-warm1))

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(TestAdamW, self)._prepare_local(var_device, var_dtype, apply_state)
        rate1 = apply_state[(var_device, var_dtype)]['lr_t']
        beta1 = tf.identity(self._get_hyper('beta_1', var_dtype))
        beta2 = tf.identity(self._get_hyper('beta_2', var_dtype))
        apply_state[(var_device, var_dtype)].update(dict(
            lr=self._rate_sch(rate1, tf.cast(self.iterations+1, var_dtype), self.step+1) if self.sch else rate1,
            epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
            beta_1=beta1,
            beta_1_minus=1-beta1,
            beta_2=beta2,
            beta_2_minus=1-beta2))

    def _resource_apply_base(self, grad, var, indices=None, apply_state=None):
        devi1, type1, name1 = var.device, var.dtype.base_dtype, var.name
        spec1 = any(c1 in name1.lower() for c1 in self.spec)
        coef1 = ((apply_state or {}).get((devi1, type1)) or self._fallback_apply_state(devi1, type1))
        m1, v1, r1 = self.get_slot(var, 'm'), self.get_slot(var, 'v'), 1.0

        if indices is None:
            m2 = m1.assign(coef1['beta_1']*m1+coef1['beta_1_minus']*grad, self._use_locking)
            v2 = v1.assign(coef1['beta_2']*v1+coef1['beta_2_minus']*grad*grad, self._use_locking)

        else:
            m2 = m1.assign(coef1['beta_1']*m1, self._use_locking)
            v2 = v1.assign(coef1['beta_2']*v1, self._use_locking)

            with tf.control_dependencies([m2, v2]):
                m2 = self._resource_scatter_add(m1, indices, coef1['beta_1_minus']*grad)
                v2 = self._resource_scatter_add(v1, indices, coef1['beta_2_minus']*grad*grad)

        u1 = m2/(tf.sqrt(v2)+coef1['epsilon'])
        u1 = u1 if spec1 else u1+self.drate*var

        if self.lmode == 1 and not spec1:
            n1, n2 = tf.norm(var, 2), tf.norm(u1, 2)
            r1 = tf.where(tf.greater(n1, 0.), tf.where(tf.greater(n2, 0.), n1/n2, 1.), 1.)

        if self.lmode == 2 and not spec1:
            r1 = self.ldecay.get(([c2 for c2 in self.ldecay.keys() if c2 in name1]+[''])[0], r1)

        return tf.group(*[var.assign_sub(r1*coef1['lr']*u1, self._use_locking), m2, v2])

    def _resource_apply_dense(self, grad, var, apply_state=None, **kwargs):
        return self._resource_apply_base(grad, var, None, apply_state)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None, **kwargs):
        return self._resource_apply_base(grad, var, indices, apply_state)

    def get_config(self):
        conf1 = super(TestAdamW, self).get_config()
        conf1.update({
            'learning_rate': self._serialize_hyperparameter('learning_rate'),
            'beta_1': self._serialize_hyperparameter('beta_1'),
            'beta_2': self._serialize_hyperparameter('beta_2'),
            'decaying_rate': self.drate,
            'epsilon': self.epsilon,
            'step': self.step})
        return conf1
