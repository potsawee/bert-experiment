import modeling
from bert_tokenizer import *
import tensorflow as tf
import numpy as np
import os
import sys
import random
import pdb

class BertClassifier(object):
    def __init__(self):
        # input_ids = [batch_size, seq_length]
        self.input_ids = tf.placeholder(tf.int32, [None, None], name="input_ids")
        self.input_mask = tf.placeholder(tf.int32, [None, None], name="input_mask")
        self.labels = tf.placeholder(tf.int32, [None], name="labels")

        # self.create_model(bert_config, is_training, self.input_ids, self.input_mask, self.labels, num_labels)
        # self.init_model(init_checkpoint)
        # self.create_optimizer()

    def create_model(self, bert_config, is_training, input_ids, input_mask, labels, num_labels):

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask)

        output_layer = model.get_pooled_output()
        hidden_size = output_layer.shape[-1].value

        with tf.variable_scope("ontop"):
            output_weights = tf.get_variable(
                "output_weights", [num_labels, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            per_example_loss = -tf.reduce_sum(one_hot_labels*log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)

            self.loss = loss
            self.probabilities = probabilities

    def init_model(self, init_checkpoint):
        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        print("*** Trainable Variables ***")
        for var in tvars:
            if var.name in initialized_variable_names:
                print("name = {}, shape = {}  <INIT_FROM_CKPT>".format(var.name, var.shape))
            else:
                print("name = {}, shape = {}".format(var.name, var.shape))

    def create_optimizer(self):
        # backpropagation
        tvars = tf.trainable_variables()
        gradients = tf.gradients(self.loss, tvars)
        max_gradient_norm = 1.0 # set to value like 1.0, 5.0
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        # optimisation
        learning_rate = 0.0005
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(clipped_gradients, tvars))

    def fine_tune(self, train_data, valid_data, num_epochs=5):
        save_path = "/home/alta/BLTSpeaking/ged-pm574/summer2019/bert102/fine-tuned"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    	# save & restore model
        saver = tf.train.Saver(max_to_keep=2)

        use_gpu = True
        if use_gpu == True:
            if 'X_SGE_CUDA_DEVICE' in os.environ:
                print('running on the stack...')
                cuda_device = os.environ['X_SGE_CUDA_DEVICE']
                print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

            else: # development only e.g. air202
                print('running locally...')
                os.environ['CUDA_VISIBLE_DEVICES'] = '3' # choose the device (GPU) here

            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True # Whether the GPU memory usage can grow dynamically.
            sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95 # The fraction of GPU memory that the process can use.

        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            sess_config = tf.ConfigProto()

        with tf.Session(config=sess_config) as sess:

            sess.run(tf.global_variables_initializer())

            batch_size = 32
            len_train_data = len(train_data)

            count_progress_step = int(len_train_data/20)

            for epoch in range(num_epochs):
                random.shuffle(train_data)
                i = 0
                train_loss = 0
                count_progress = count_progress_step

                while i < len_train_data:
                    j = 0
                    input_ids = []
                    input_mask = []
                    labels = []
                    while j < batch_size and i+j < len_train_data:
                        input_ids.append(train_data[i+j].input_ids)
                        input_mask.append(train_data[i+j].input_mask)
                        labels.append(train_data[i+j].label_id)
                        j += 1



                    feed_dict = {self.input_ids: input_ids,
                                self.input_mask: input_mask,
                                self.labels: labels}

                    [_, loss] = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
                    train_loss += loss
                    i += batch_size
                    if i > count_progress:
                        print('#', end='')
                        count_progress += count_progress_step
                        sys.stdout.flush()
                print("")
                print("epoch {}: train loss = {}".format(epoch, train_loss/len(train_data)))

                # --------------------------- compute validation loss --------------------------- #
                vi = 0
                valid_loss = 0
                while vi < len(valid_data):
                    vj = 0
                    valid_input_ids = []
                    valid_input_mask = []
                    valid_labels = []
                    while vj < batch_size:
                        valid_input_ids.append(valid_data[vi+vj].input_ids)
                        valid_input_mask.append(valid_data[vi+vj].input_mask)
                        valid_labels.append(valid_data[vi+vj].label_id)
                        vj += 1

                    valid_feed_dict = {self.input_ids: valid_input_ids,
                                      self.input_mask: valid_input_mask,
                                      self.labels: valid_labels}

                    a_loss = sess.run(self.loss, feed_dict=valid_feed_dict)
                    valid_loss += a_loss
                    vi += batch_size
                print("epoch {}: validation loss = {}".format(epoch, valid_loss/len(valid_data)))
                # -------------------------------------------------------------------------------- #

                saver.save(sess, save_path + "/model", global_step=epoch)

    def predict(self, data, saved_model):
        scores_avg = []
        scores_argmax = []

        # 0. set up environment
        use_gpu = True
        if use_gpu == True:
            if 'X_SGE_CUDA_DEVICE' in os.environ:
                print('running on the stack...')
                cuda_device = os.environ['X_SGE_CUDA_DEVICE']
                print('X_SGE_CUDA_DEVICE is set to {}'.format(cuda_device))
                os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

            else: # development only e.g. air202
                print('running locally...')
                os.environ['CUDA_VISIBLE_DEVICES'] = '3' # choose the device (GPU) here

            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True # Whether the GPU memory usage can grow dynamically.
            sess_config.gpu_options.per_process_gpu_memory_fraction = 0.95 # The fraction of GPU memory that the process can use.

        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            sess_config = tf.ConfigProto()
            
        with tf.Session(config=sess_config) as sess:

            # 1. load a trained (fine-tuned) model
            saver = tf.train.Saver()
            saver.restore(sess, saved_model)
            print('loaded model...', saved_model)

            # 2. make prediction
            num_features = len(data)
            batch_size = 1000
            i = 0
            while i < num_features:
                j = 0
                input_ids = []
                input_mask = []
                while j < batch_size and i+j < num_features:
                    input_ids.append(data[i+j].input_ids)
                    input_mask.append(data[i+j].input_mask)
                    j += 1

                feed_dict = {self.input_ids: input_ids, self.input_mask: input_mask}

                probs = sess.run(self.probabilities, feed_dict=feed_dict)

                for x in probs:
                    score_avg = x[0]*0 + x[1]*1 + x[2]*2 + x[3]*3 + x[4]*4
                    score_argmax = np.argmax(x)
                    scores_avg.append(score_avg)
                    scores_argmax.append(score_argmax)

                i += batch_size
                print('#', end='')
                sys.stdout.flush()

        return scores_avg, scores_argmax


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, label_id,is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.is_real_example = is_real_example

def prepare_sentiment_data(data_file, vocab_file, do_lower_case, max_seq_length=64):
    features = [] # a list of features (training instances)

    phrases, sentiments = read_movie_reviews(data_file)
    bert_tokenizer = BertTokenizer(vocab_file, do_lower_case)
    for phrase, sentiment in zip(phrases, sentiments):
        tokens = bert_tokenizer.raw2tokens(phrase)

        if len(tokens) > max_seq_length-2: # to account for [CLS] and [SEP]
            tokens = tokens[:max_seq_length-2]

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = bert_tokenizer.tokens2ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1]*len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        label_id = int(sentiment)

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            label_id=label_id,
            is_real_example=True)

        features.append(feature)
    return features

def prepare_sentiment_test(data_file, vocab_file, do_lower_case, max_seq_length=64):
	features = []
	phrases, phrase_ids = read_movie_test(data_file)
	bert_tokenizer = BertTokenizer(vocab_file, do_lower_case)
	for phrase in phrases:
		tokens = bert_tokenizer.raw2tokens(phrase)

		if len(tokens) > max_seq_length-2: # to account for [CLS] and [SEP]
			tokens = tokens[:max_seq_length-2]

		tokens = ['[CLS]'] + tokens + ['[SEP]']
		input_ids = bert_tokenizer.tokens2ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
		input_mask = [1]*len(input_ids)

		while len(input_ids) < max_seq_length:
			input_ids.append(0)
			input_mask.append(0)
		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length

		feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            label_id=None,
            is_real_example=True)

		features.append(feature)

	return features, phrase_ids

def main():
    # config
    bert_config_file = "/home/alta/BLTSpeaking/ged-pm574/summer2019/bert_pretrained/uncased_L-12_H-768_A-12/bert_config.json"
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    init_checkpoint = "/home/alta/BLTSpeaking/ged-pm574/summer2019/bert_pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt"
    num_labels = 5

    data_file = "/home/alta/BLTSpeaking/ged-pm574/summer2019/sentiment1_data/train.tsv"
    test_file = "/home/alta/BLTSpeaking/ged-pm574/summer2019/sentiment1_data/test.tsv"
    vocab_file = "/home/alta/BLTSpeaking/ged-pm574/summer2019/bert_pretrained/uncased_L-12_H-768_A-12/vocab.txt"

    features = prepare_sentiment_data(data_file, vocab_file, do_lower_case=True)
    test_features, phrase_ids = prepare_sentiment_test(test_file, vocab_file, do_lower_case=True)
    random.seed(25)
    random.shuffle(features)
    train_data = features[960:]
    valid_data = features[:960]
    print("finish loading data")

    mode = 'fine_tune'

    if mode == 'fine_tune':
        classifer = BertClassifier()
        is_training = True
        classifer.create_model(bert_config, is_training, classifer.input_ids, classifer.input_mask, classifer.labels, num_labels)
        classifer.init_model(init_checkpoint)
        classifer.create_optimizer()
        classifer.fine_tune(train_data=train_data, valid_data=valid_data, num_epochs=5)
    elif mode == 'predict':
        classifer = BertClassifier()
        is_training = False
        classifer.create_model(bert_config, is_training, classifer.input_ids, classifer.input_mask, classifer.labels, num_labels)
        saved_model = "/home/alta/BLTSpeaking/ged-pm574/summer2019/bert102/fine-tuned/model-4"
        scores_avg,  scores_argmax = classifer.predict(data=test_features, saved_model=saved_model)

        assert len(phrase_ids) == len(scores_avg)
        assert len(phrase_ids) == len(scores_argmax)

        with open("predictions_avg.csv", 'w') as file:
            for id, score in zip(phrase_ids, scores_avg):
                grade = int(round(score))
                file.write("{},{}\n".format(id, grade))

        with open("predictions_argmax.csv", 'w') as file:
            for id, score in zip(phrase_ids, scores_argmax):
                grade = int(round(score))
                file.write("{},{}\n".format(id, grade))

    else:
        raise ValueError("Mode not recognised")
# def dev():
# 	pdb.set_trace()
#	test_file = "/home/alta/BLTSpeaking/ged-pm574/summer2019/sentiment1_data/test.tsv"
#	vocab_file = "/home/alta/BLTSpeaking/ged-pm574/summer2019/bert_pretrained/uncased_L-12_H-768_A-12/vocab.txt"
#	test_features, phrase_ids = prepare_sentiment_test(test_file, vocab_file, do_lower_case=True)


if __name__ == "__main__":
    main()
