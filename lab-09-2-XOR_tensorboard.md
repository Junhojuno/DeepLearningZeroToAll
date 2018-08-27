
### TensorBoard with 5 step
1. From TF graph, decide which tensors you wants to log
    - ex) w2_hist = tf.summary.histogram("weights2", w2)
    - ex) cost_summary = tf.summary.scalar("cost", cost)
    - tf.name_scope로 hierarchical정리 가능
    
2. Merge all summaries
    - ex) summary = tf.summary.merge_all()
    
3. Create writer and add graph
    - ex) writer = tf.summary.FileWriter('./logs') # file위치 지정
    - ex) writer.add_graph(sess.graph)

4. Run summary merge and add_summary
    - ex) s, _ = sess.run([summary, optimizer], feed_dict=feed_dict)
    - ex) writer.add_summary(s, global_step=global_step)

5. Launch TensorBoard
    : `$ tensorboard -logdir=./logs/xor_logs`


```python
# Lab 9 XOR
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.01

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

# set X, Y
X = tf.placeholder(tf.float32, [None, 2], name='x-input')
Y = tf.placeholder(tf.float32, [None, 1], name='y-input')
```


```python
# set weight and bias in multilayer
# set summary
with tf.name_scope("layer1"):
    W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
    b1 = tf.Variable(tf.random_normal([2]), name='bias1')
    layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)


with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
    b2 = tf.Variable(tf.random_normal([1]), name='bias2')
    hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)
```


```python
# cost/loss function
with tf.name_scope("cost"):
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                           tf.log(1 - hypothesis))
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("train"):
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
```


```python
# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# Launch graph
with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_01")
    writer.add_graph(sess.graph)  # Show the graph

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        summary, _ = sess.run([merged_summary, train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)

        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run([W1, W2]))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
```

    0 0.7156377 [array([[ 0.7926959 ,  0.6886104 ],
           [-1.2072834 , -0.29517072]], dtype=float32), array([[1.7177    ],
           [0.35572484]], dtype=float32)]
    1000 0.022333965 [array([[ 5.6736493, -6.2197533],
           [-6.14471  ,  6.572153 ]], dtype=float32), array([[8.90876 ],
           [8.472787]], dtype=float32)]
    2000 0.006309393 [array([[ 6.6223445, -7.192567 ],
           [-7.0854316,  7.526798 ]], dtype=float32), array([[11.223157],
           [10.811347]], dtype=float32)]
    3000 0.0027709822 [array([[ 7.1489196, -7.732995 ],
           [-7.6093445,  8.058455 ]], dtype=float32), array([[12.759906],
           [12.362965]], dtype=float32)]
    4000 0.0014247245 [array([[ 7.5337973, -8.12768  ],
           [-7.9929643,  8.447399 ]], dtype=float32), array([[14.0166645],
           [13.630341 ]], dtype=float32)]
    5000 0.0007894051 [array([[ 7.849699, -8.451324],
           [-8.308202,  8.766735]], dtype=float32), array([[15.141127],
           [14.763162]], dtype=float32)]
    6000 0.00045509648 [array([[ 8.125374, -8.733495],
           [-8.583534,  9.045429]], dtype=float32), array([[16.19618 ],
           [15.825219]], dtype=float32)]
    7000 0.0002683317 [array([[ 8.374636, -8.988396],
           [-8.83265 ,  9.297394]], dtype=float32), array([[17.213171],
           [16.848274]], dtype=float32)]
    8000 0.00016026004 [array([[ 8.604925, -9.223699],
           [-9.062923,  9.530158]], dtype=float32), array([[18.209024],
           [17.849438]], dtype=float32)]
    9000 9.648972e-05 [array([[ 8.820601, -9.443915],
           [-9.278695,  9.748132]], dtype=float32), array([[19.193117],
           [18.838554]], dtype=float32)]
    10000 5.8339763e-05 [array([[ 9.024393, -9.651846],
           [-9.482652,  9.95404 ]], dtype=float32), array([[20.17106 ],
           [19.820868]], dtype=float32)]
    
    Hypothesis:  [[6.1310318e-05]
     [9.9993694e-01]
     [9.9995077e-01]
     [5.9751477e-05]] 
    Correct:  [[0.]
     [1.]
     [1.]
     [0.]] 
    Accuracy:  1.0
    

### Multiple runs


```python
# cost/loss function
with tf.name_scope("cost"):
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                           tf.log(1 - hypothesis))
    cost_summ = tf.summary.scalar("cost", cost)

with tf.name_scope("train"):
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
```


```python
# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar("accuracy", accuracy)

# Launch graph
with tf.Session() as sess:
    # tensorboard --logdir=./logs/xor_logs
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/xor_logs_r0_1")
    writer.add_graph(sess.graph)  # Show the graph

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        summary, _ = sess.run([merged_summary, train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)

        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run([W1, W2]))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)
```

    0 0.71064335 [array([[ 0.70269835,  0.77858704],
           [-1.117285  , -0.20518988]], dtype=float32), array([[1.8076982 ],
           [0.44572306]], dtype=float32)]
    1000 0.0006704356 [array([[-8.57351 ,  9.078539],
           [ 8.603754, -9.359739]], dtype=float32), array([[15.069714],
           [15.171516]], dtype=float32)]
    2000 0.00019878645 [array([[-9.120513,  9.596931],
           [ 9.149157, -9.879135]], dtype=float32), array([[17.442375],
           [17.550861]], dtype=float32)]
    3000 8.798035e-05 [array([[ -9.452483,   9.914254],
           [  9.480015, -10.197143]], dtype=float32), array([[19.039598],
           [19.151682]], dtype=float32)]
    4000 4.5255856e-05 [array([[ -9.705619,  10.157407],
           [  9.73226 , -10.440843]], dtype=float32), array([[20.345488],
           [20.460047]], dtype=float32)]
    5000 2.5004463e-05 [array([[ -9.919293,  10.363375],
           [  9.945173, -10.647294]], dtype=float32), array([[21.51126 ],
           [21.627794]], dtype=float32)]
    6000 1.4379724e-05 [array([[-10.109775,  10.547488],
           [ 10.134968, -10.83185 ]], dtype=float32), array([[22.602282],
           [22.720455]], dtype=float32)]
    7000 8.463896e-06 [array([[-10.28503 ,  10.717269],
           [ 10.309588, -11.002054]], dtype=float32), array([[23.651407],
           [23.770967]], dtype=float32)]
    8000 5.0515064e-06 [array([[-10.449357,  10.876779],
           [ 10.473308, -11.161944]], dtype=float32), array([[24.676378],
           [24.797178]], dtype=float32)]
    9000 3.0249403e-06 [array([[-10.605251,  11.028317],
           [ 10.628672, -11.313904]], dtype=float32), array([[25.687412],
           [25.809067]], dtype=float32)]
    10000 1.7881409e-06 [array([[-10.754226,  11.173413],
           [ 10.777166, -11.459352]], dtype=float32), array([[26.690197],
           [26.812716]], dtype=float32)]
    
    Hypothesis:  [[1.9362387e-06]
     [9.9999821e-01]
     [9.9999845e-01]
     [1.9069422e-06]] 
    Correct:  [[0.]
     [1.]
     [1.]
     [0.]] 
    Accuracy:  1.0
    
