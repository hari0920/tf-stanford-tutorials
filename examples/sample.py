import tensorflow as tf

a = tf.constant([2,2],name="a")
b = tf.constant([[3,4],[2,0]],name="b")
x = tf.add(a,b,name="addition")
y = tf.multiply(a,b,name="mul")
z = tf.matmul([a],b, name="matmul")
pow = tf.pow(x,y,name="power")
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./graphs/sample", sess.graph)
    pow = sess.run(pow)
    #p = sess.run(pow)
    print(pow,x,y)
writer.close()

"""
with tf.device('/gpu:0'):
 a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
 b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
 c = tf.multiply(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print (sess.run(c))
"""