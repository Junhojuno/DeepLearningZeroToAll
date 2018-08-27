

```python
# Lab 9 XOR
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1

x_data = np.array([[0, 0],[0, 1],[1, 0],[1, 1]], dtype=np.float32)
y_data = np.array([[0],
          [1],
          [1],
          [0]], dtype=np.float32)
```

####  XOR with Logistic Regression


```python
# set Variables
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
```


```python
# set hypotheis
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)
```


```python
# set cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
```


```python
# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
```


```python
# launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ",h, "\nPredicted: ",p, "\nAccuracy:",a)
```

    0 0.87597954
    1000 0.6931492
    2000 0.6931472
    3000 0.6931472
    4000 0.6931472
    5000 0.6931472
    6000 0.6931472
    7000 0.6931472
    8000 0.6931472
    9000 0.6931472
    10000 0.6931472
    
    Hypothesis:  [[0.5]
     [0.5]
     [0.5]
     [0.5]] 
    Predicted:  [[0.]
     [0.]
     [0.]
     [0.]] 
    Accuracy: 0.5
    

##### tf.cast함수
텐서를 새로운 형태로 캐스팅하는데 사용한다.

부동소수점형에서 정수형으로 바꾼 경우 소수점 버린을 한다.

Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

### 아무리 해도 0.5밖에 안나온다.--> Neural Net활용


```python
W1 = tf.Variable(tf.random_normal([2,2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X ,W1) + b1)

W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2)+b2)
```


```python
# set cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
```


```python
# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
```


```python
# launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ",h, "\nPredicted: ",p, "\nAccuracy:",a)
```

    0 1.1806753
    1000 0.69142807
    2000 0.6849137
    3000 0.6303352
    4000 0.51307726
    5000 0.27579618
    6000 0.08628263
    7000 0.047703177
    8000 0.03253933
    9000 0.0245574
    10000 0.01966428
    
    Hypothesis:  [[0.02277964]
     [0.9835014 ]
     [0.9812571 ]
     [0.01985719]] 
    Predicted:  [[0.]
     [1.]
     [1.]
     [0.]] 
    Accuracy: 1.0
    

### Wide Neural Net for XOR
- 2개 입력받고, 10개를 출력(layer1)
- 이 10개를 다시 입력받아 1개 출력(layer2)
- 어떤 결과가 나올 것인가?


```python
W1 = tf.Variable(tf.random_normal([2,10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X ,W1) + b1)

W2 = tf.Variable(tf.random_normal([10,1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2)+b2)
```


```python
# set cost function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
```


```python
# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
```


```python
# launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
    h, p, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis: ",h, "\nPredicted: ",p, "\nAccuracy:",a)
```

    0 1.894959
    1000 0.39848888
    2000 0.123508096
    3000 0.05041793
    4000 0.028694328
    5000 0.019343074
    6000 0.014338989
    7000 0.011281742
    8000 0.009242748
    9000 0.0077960747
    10000 0.0067214603
    
    Hypothesis:  [[0.00513254]
     [0.9941058 ]
     [0.99201715]
     [0.0077831 ]] 
    Predicted:  [[0.]
     [1.]
     [1.]
     [0.]] 
    Accuracy: 1.0
    

0인것은 좀 더 0에 가깝게 작아졌고, y=1인건 1에 가깝게 더 커졌다
즉 ,좀 더 정교해졌다고 볼 수 있다.
