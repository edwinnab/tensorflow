import tensorflow as tf
#variable creation
my_var = tf.Variable(2)
#constant
my_constant = tf.constant(3)
new_value = tf.add(my_var,my_constant)
#changing value of a variable
new_var = tf.assign(my_var, new_value)