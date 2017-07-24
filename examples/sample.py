import tensorflow as tf
a = tf.constant([2,2],name="a")
b = tf.constant([[3,4],[2,0]],name="b")
x = tf.add(a,b,name="add")
y = tf.multiply(a,b,name='mul')
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./graphs/sample", sess.graph)
    x,y = sess.run([x,y])
    print(x,y)
writer.close()