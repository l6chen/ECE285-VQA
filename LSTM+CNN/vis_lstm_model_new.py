import tensorflow as tf
import math
import numpy as np

class Vis_lstm_model:
	def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
		return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

	def init_bias(self, dim_out, name=None):
		return tf.Variable(tf.zeros([dim_out]), name=name)

	def __init__(self, options):
		with tf.device('/gpu:0'):
			self.options = options

			# +1 for zero padding
			self.Wemb = tf.Variable(tf.random_uniform([options['q_vocab_size'] + 1, options['embedding_size']], -1.0, 1.0), name = 'Wemb')
			self.Wimg = self.init_weight(options['fc7_feature_length'], options['embedding_size'], name = 'Wimg')
			self.bimg = self.init_bias(options['embedding_size'], name = 'bimg')
			
			#Initialize model
# 			self.lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units = options['rnn_size'],
#                                                      dtype = tf.float32, 
#                                                      name = 'lstm')
			num_units = [options['rnn_size']] * options['lstm_steps']
			self.lstm = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers = 2,
                                                        num_units = 512,
                                                        input_mode='linear_input',
                                                        direction='unidirectional',
                                                        dropout=0.0,
                                                        seed=None,
                                                        dtype=tf.float32,
                                                        kernel_initializer=None,
                                                        bias_initializer=None,
                                                        name='lstm'
                                                    ) 
# 			cells = [tf.nn.rnn_cell.LSTMCell(num_units=n) for n in num_units]

# 			self.lstm = tf.nn.rnn_cell.MultiRNNCell(cells)
            
			self.ans_sm_W = self.init_weight(options['rnn_size'], options['ans_vocab_size'], name = 'ans_sm_W')
			self.ans_sm_b = self.init_bias(options['ans_vocab_size'], name = 'ans_sm_b')


	def forward_pass_lstm(self, word_embeddings):

		x = tf.stack(word_embeddings,0)
		#print(x.shape)
# 		for i in range(self.options['lstm_steps']):
# 			if i == 0:
# 				outputs, state = tf.nn.dynamic_rnn(cell=self.lstm,
#                                                    inputs=x,
#                                                    dtype=tf.float32,time_major=True)
# 			else:
# 				outputs, state = tf.nn.dynamic_rnn(cell=self.lstm,
#                                                    inputs=outputs,
#                                                    initial_state=state,
#                                                    dtype=tf.float32,time_major=True)                
		outputs = self.lstm(x)
		print(len(outputs))
		print(outputs)
		#assert outputs is int
		return outputs[0]



	def build_model(self):
		fc7_features = tf.placeholder('float32',[None, self.options['fc7_feature_length'] ], name = 'fc7')
		sentence = tf.placeholder('int32',[None, self.options['lstm_steps'] - 1], name = "sentence")#is a matrix[ques,512]
		answer = tf.placeholder('float32', [None, self.options['ans_vocab_size']], name = "answer")


		word_embeddings = []
		for i in range(self.options['lstm_steps']-1):
			word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,i])
			word_emb = tf.nn.dropout(word_emb, self.options['word_emb_dropout'], name = "word_emb" + str(i))
			
			word_embeddings.append(word_emb)

		image_embedding = tf.matmul(fc7_features, self.Wimg) + self.bimg
		image_embedding = tf.nn.tanh(image_embedding)
		image_embedding = tf.nn.dropout(image_embedding, self.options['image_dropout'], name = "vis_features")

		# Image as the last word in the lstm
		word_embeddings.append(image_embedding)
# 		word_embeddings = tf.convert_to_tensor(word_embeddings)
# 		print(word_embeddings.shape)
		lstm_output = self.forward_pass_lstm(word_embeddings)
		lstm_answer = lstm_output[-1,:,:]
		print(lstm_answer)
		logits = tf.matmul(lstm_answer, self.ans_sm_W) + self.ans_sm_b
		# ce = tf.nn.softmax_cross_entropy_with_logits(logits, answer, name = 'ce')
		ce = tf.nn.softmax_cross_entropy_with_logits(labels=answer, logits= logits, name = 'ce')
		answer_probab = tf.nn.softmax(logits, name='answer_probab')
		
		predictions = tf.argmax(answer_probab,1)
		correct_predictions = tf.equal(tf.argmax(answer_probab,1), tf.argmax(answer,1))
		accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

		loss = tf.reduce_sum(ce, name = 'loss')
		input_tensors = {
			'fc7' : fc7_features,
			'sentence' : sentence,
			'answer' : answer
		}
		return input_tensors, loss, accuracy, predictions
    
	def build_generator(self):
		fc7_features = tf.placeholder('float32',[ None, self.options['fc7_feature_length'] ], name = 'fc7')
		sentence = tf.placeholder('int32',[None, self.options['lstm_steps'] - 1], name = "sentence")

		word_embeddings = []
		for i in range(self.options['lstm_steps']-1):
			word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,i])
			word_embeddings.append(word_emb)

		image_embedding = tf.matmul(fc7_features, self.Wimg) + self.bimg
		image_embedding = tf.nn.tanh(image_embedding)

		word_embeddings.append(image_embedding)
		lstm_output = self.forward_pass_lstm(word_embeddings)
		lstm_answer = lstm_output[-1]
		logits = tf.matmul(lstm_answer, self.ans_sm_W) + self.ans_sm_b
		
		answer_probab = tf.nn.softmax(logits, name='answer_probab')
		
		predictions = tf.argmax(answer_probab,1)

		input_tensors = {
			'fc7' : fc7_features,
			'sentence' : sentence
		}

		return input_tensors, predictions, answer_probab