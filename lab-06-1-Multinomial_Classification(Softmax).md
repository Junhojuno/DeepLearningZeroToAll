
### TensorFlow로 Softmax Classification의 구현하기
- Hypothesis
- Cost function : Cross entropy 
- Optimize, Minimize


```python
import tensorflow as tf
```


```python
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1], # one-hot encoding
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
```


```python
X = tf.placeholder(tf.float32, shape=[None,4])
Y = tf.placeholder(tf.float32, shape=[None, 3])
nb_classes = 3 # label(class) : 3개

w = tf.Variable(tf.random_normal([4,3]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
```


```python
# set hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X,w)+b)
```


```python
# Cost : Cross entropy cost/loss
cost = tf.reduce_mean(- tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
```


```python
# optimize and minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
```


```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
```

    0 4.2193084
    200 0.7190609
    400 0.62087977
    600 0.5538566
    800 0.49731112
    1000 0.44549114
    1200 0.39486104
    1400 0.34241518
    1600 0.2942958
    1800 0.275319
    2000 0.25928077
    


```python
# 적용
sess = tf.Session()
sess.run(tf.global_variables_initializer())
a = sess.run(hypothesis, feed_dict={X: [[1,11,7,9]]})
print(a, sess.run(tf.argmax(a, 1)))
```

    [[1.7461476e-04 9.9982542e-01 9.5283451e-12]] [1]
    


```python
sess.close()
```
