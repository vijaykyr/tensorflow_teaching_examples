import tensorflow as tf

#Define input feature columns
sq_footage = tf.contrib.layers.real_valued_column("sq_footage")
feature_columns = [sq_footage]

#Define input function
def input_fn(feature_data,label_data=None):
  return {"sq_footage":feature_data}, label_data
	
#Instantiate Linear Regression Model
estimator = tf.contrib.learn.LinearRegressor(
  feature_columns=feature_columns,
  optimizer=tf.train.FtrlOptimizer(learning_rate=100))

#Train
estimator.fit(
  input_fn=lambda:input_fn(tf.constant([1000,2000]),
                           tf.constant([100000,200000])),
  steps=100)

#Predict
estimator.predict(input_fn=lambda: input_fn(tf.constant([3000])))