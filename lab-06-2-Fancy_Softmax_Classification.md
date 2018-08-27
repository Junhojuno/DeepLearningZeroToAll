
### Fancy Softmax Classifier
- 좀 더 예쁘게 softmax 작성
- cross entropy : `tf.nn.softmax_cross_entropy_with_logits()`
    - logits = `tf.matmul(X,w) + b` : softmax에 들어가기 전 값
    - hypothesis = `tf.nn.sortmax(logits)` : softmax에 들어간 후 값
- one-hot encoding
- reshape 


```python
import tensorflow as tf
```

##### Animal Classification


```python
xy = np.loadtxt("data-04-zoo.csv", delimiter=",", dtype=np.float32)
x_data = xy[:,0:-1]
y_data = xy[:,[-1]]
```


```python
# Y one-hot encoding
nb_classes = 7
Y = tf.placeholder(tf.int32, shape=[None, 1]) # 0~6, shape=(?,1)
Y_one_hot = tf.one_hot(Y, nb_classes) # one hot, shape=(?,1,7)
# 차원(rank)이 하나 더 만들어져서 이를 없애주기 위해 reshape
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # shape=(?,7), -1:everything 
```


```python
X = tf.placeholder(tf.float32, shape=[None, 16])
w = tf.Variable(tf.random_normal([16,nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')
```


```python
# set logits and hypothesis
logits = tf.matmul(X,w) + b
hypothesis = tf.nn.softmax(logits)
```


```python
# set cost function
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
```


```python
# predict
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```


```python
# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2000):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X:x_data, Y:y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))
            
    pred = sess.run(prediction, feed_dict={X:x_data})
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))
```

    Step:     0	Loss: 4.504	Acc: 31.68%
    Step:   100	Loss: 0.557	Acc: 90.10%
    Step:   200	Loss: 0.378	Acc: 90.10%
    Step:   300	Loss: 0.294	Acc: 92.08%
    Step:   400	Loss: 0.243	Acc: 94.06%
    Step:   500	Loss: 0.208	Acc: 95.05%
    Step:   600	Loss: 0.182	Acc: 97.03%
    Step:   700	Loss: 0.161	Acc: 97.03%
    Step:   800	Loss: 0.145	Acc: 97.03%
    Step:   900	Loss: 0.131	Acc: 97.03%
    Step:  1000	Loss: 0.120	Acc: 97.03%
    Step:  1100	Loss: 0.110	Acc: 97.03%
    Step:  1200	Loss: 0.101	Acc: 99.01%
    Step:  1300	Loss: 0.094	Acc: 99.01%
    Step:  1400	Loss: 0.087	Acc: 99.01%
    Step:  1500	Loss: 0.081	Acc: 99.01%
    Step:  1600	Loss: 0.076	Acc: 99.01%
    Step:  1700	Loss: 0.071	Acc: 100.00%
    Step:  1800	Loss: 0.067	Acc: 100.00%
    Step:  1900	Loss: 0.063	Acc: 100.00%
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 6 True Y: 6
    [True] Prediction: 6 True Y: 6
    [True] Prediction: 6 True Y: 6
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 5 True Y: 5
    [True] Prediction: 4 True Y: 4
    [True] Prediction: 4 True Y: 4
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 5 True Y: 5
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 5 True Y: 5
    [True] Prediction: 5 True Y: 5
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 5 True Y: 5
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 6 True Y: 6
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 5 True Y: 5
    [True] Prediction: 4 True Y: 4
    [True] Prediction: 6 True Y: 6
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 2 True Y: 2
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 6 True Y: 6
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 2 True Y: 2
    [True] Prediction: 6 True Y: 6
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 2 True Y: 2
    [True] Prediction: 6 True Y: 6
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 6 True Y: 6
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 5 True Y: 5
    [True] Prediction: 4 True Y: 4
    [True] Prediction: 2 True Y: 2
    [True] Prediction: 2 True Y: 2
    [True] Prediction: 3 True Y: 3
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 1 True Y: 1
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 5 True Y: 5
    [True] Prediction: 0 True Y: 0
    [True] Prediction: 6 True Y: 6
    [True] Prediction: 1 True Y: 1
    
