import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):

        # Linearly increasing the learning rate for the first warmup_steps then decrease
        step_num = tf.cast(step_num, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step_num)
        arg2 = step_num * (self.warmup_steps ** -1.5)

        return 1./tf.math.sqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    