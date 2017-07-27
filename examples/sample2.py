import tensorflow as tf
graph = tf.Graph()
with graph.as_default():
  variable = tf.Variable(42, name='foo')
  assign = variable.assign(13)
  initialize = tf.global_variables_initializer()


with tf.Session(graph=graph) as sess:
  sess.run(initialize)
  #sess.run(assign)
  print(sess.run(variable))

with tf.Session(graph=graph) as sess:
     sess.run(initialize)
     sess.run(assign)
     print(sess.run(variable))