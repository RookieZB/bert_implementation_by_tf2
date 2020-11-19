# -*- coding: utf-8 -*-

"""
AdaBelief Optimizer with Weight Decay (not sure if is correct)
https://github.com/juntang-zhuang/Adabelief-Optimizer

"""

import tensorflow as tf
import mymodels as mm


class AdaBelief(mm.AdamW):
    def __init__(self, **kwargs):
        super(AdaBelief, self).__init__(**kwargs)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'm')
            self.add_slot(var, 'v')
            self.add_slot(var, 'g')

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdaBelief, self)._prepare_local(var_device, var_dtype, apply_state)
        step1 = tf.cast(self.iterations+1, var_dtype)
        apply_state[(var_device, var_dtype)].update(dict(
            beta_1_power=tf.pow(tf.identity(self._get_hyper('beta_1', var_dtype)), step1),
            beta_2_power=tf.pow(tf.identity(self._get_hyper('beta_2', var_dtype)), step1)))

    def _resource_apply_base(self, grad, var, indices=None, apply_state=None):
        devi1, type1, name1 = var.device, var.dtype.base_dtype, var.name
        spec1 = any(c1 in name1.lower() for c1 in self.spec)
        coef1 = ((apply_state or {}).get((devi1, type1)) or self._fallback_apply_state(devi1, type1))
        m1, v1, g1, r1 = self.get_slot(var, 'm'), self.get_slot(var, 'v'), self.get_slot(var, 'g'), 1.0

        if indices is None:
            m2 = m1.assign(coef1['beta_1']*m1+coef1['beta_1_minus']*grad, self._use_locking)
            g2, g3 = g1.assign(grad-m2), coef1['beta_2']*v1
            v2 = v1.assign(g3+coef1['beta_2_minus']*g2*g2, self._use_locking)

        else:
            m2 = m1.assign(coef1['beta_1']*m1, self._use_locking)

            with tf.control_dependencies([m2]):
                m2 = self._resource_scatter_add(m1, indices, coef1['beta_1_minus']*grad)
                g2 = g1.assign(-1.*m2)

            with tf.control_dependencies([g2]):
                g2 = self._resource_scatter_add(g1, indices, grad)
                v2 = v1.assign(coef1['beta_2']*v1+coef1['beta_2_minus']*g2*g2+coef1['epsilon'], self._use_locking)

        u1 = (m2/(1-coef1['beta_1_power']))/(tf.sqrt(v2/(1-coef1['beta_2_power']))+coef1['epsilon'])
        u1 = u1 if spec1 else u1+self.drate*var
        return tf.group(*[var.assign_sub(r1*coef1['lr']*u1, self._use_locking), m2, v2, g2])
