import tensorflow as tf
import os
import collections
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense,  Embedding, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.constraints import max_norm
from keras.initializers import RandomUniform
from matplotlib import pyplot as plt

#TODO: run on google COLAB
#TODO: add env variables (use pretrained model or not, use dropout, use GRU or LSTM)


DATA_PATH = './data'
MODEL_PATH = './pretrained_models'
USE_GRU = False
USE_DROPOUT = True

def read_words(filename):
    """
    reads file
    :param filename: str, location of file
    :return: lst
    """
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    """
    creates mapping from word to integer
    :param filename: str, location of file
    :return: dict <key: str, value: int>
    """
    lst = read_words(filename)
    counter = collections.Counter(lst)
    word_to_idx = dict(zip(counter.keys(), range(len(counter))))
    return word_to_idx


def file_to_word_ids(filename, word_to_id):
    """
    reads file and converts each word in the file to the matching index,
    and makes sure there is no words in the files that are not in the mapping
    :param filename: str, location of file
    :param word_to_id: dict <key: str, value: int>
    :return:list
    """
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]



def load_data():
    """
    reads train , validation and test files, creates mapping word--> idx,
    replaces the words in the files with the matching indexes from the mapping,
    creates the reverse mapping idx --> word, adds vocab size
    :return: list
    :return: list
    :return: list
    :return: int
    :return dict <key: int, value: str>
    """
    # get the data paths
    train_path = os.path.join(DATA_PATH, "ptb.train.txt")
    valid_path = os.path.join(DATA_PATH, "ptb.valid.txt")
    test_path = os.path.join(DATA_PATH, "ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_idx = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_idx)
    valid_data = file_to_word_ids(valid_path, word_to_idx)
    test_data = file_to_word_ids(test_path, word_to_idx)
    v = len(word_to_idx)
    idx_to_word = dict(zip(word_to_idx.values(), word_to_idx.keys()))

    return train_data, valid_data, test_data, v, idx_to_word



class KerasBatchGenerator(object):


    def __init__(self, data, num_steps, batch_size, v, skip_step=1):
        """

        :param data: list of integers
        :param num_steps: int , represents time dimension
        :param batch_size: int
        :param v: int , voacb size
        :param skip_step: int is the number of words which will be skipped before the next row.
               Example: Time dim =2 , words = ['hello' ,'my','name' ,'is' ,'david', 'cohen'] skip_step = 2
               Result: [['hello', 'my'] , ['name, 'is'], ['david', 'cohen']]
        """
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.v = v
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        self.skip_step = skip_step

    def generate(self):
        """

        :yields: iterator : (x np.array [batch_size, num_steps], y np.array [batch_size, vocab_size])
        """


        x = np.zeros((self.batch_size, self.num_steps)) #input fed to Embedding() layer must be [batch_size, time_dim]. After Embedding it will become 3D for LSTM
        y = np.zeros((self.batch_size,  self.v)) #we only output the next word, so return_sequence=False. Output is [batch_size, number of possible words]
        while True:
            for i in range(self.batch_size): # i is a row in our data
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set if we pass the end of the list
                    self.current_idx = 0
                x[i, :] = self.data[self.current_idx:self.current_idx + self.num_steps] #get [X(t-n), X(t-n+1), ... X(t-1), X(t)]
                temp_y = self.data[self.current_idx + self.num_steps] #get [X(t+1)] = Y
                # convert temp_y into a one hot representation
                y[i, :] = to_categorical(temp_y, num_classes=self.v)
                self.current_idx += self.skip_step


            yield x, y

def check_data_gen(data,v):
    """
    Creates one iteration from the generator. Check if X is correctly shifted by "skip_step"
    Check that Y is indeed the next word
    :param data: list of integers
    :param v: int , voacb size
    :return:
    """
    test_gen = KerasBatchGenerator(data, num_steps=5, batch_size=10, v=v, skip_step=1)
    print('hi the original data (first 50 indices ) is ')
    print(data[0:50])
    x, y = next(test_gen.generate())
    print('this is a batch of size : ', batch_size, ' and time dim of ', num_steps)
    print('X is ')
    print(x)
    print('y is ')
    print(y)
    print(' argmax of y is ')
    print(np.argmax(y, axis=1))

def plot_results(loss, val_loss,  epochs, suffix=None):
    """ Saves plot of convergence graph

    :param loss: lst
    :param val_loss: lst
    :param epochs: int
    :param suffix: str
    :return:
    """
    num_epochs = np.arange(1,epochs+1)
    plt.figure(dpi=200)
    plt.style.use('ggplot')
    plt.plot(num_epochs, loss, label='train_perplexity', c='red')
    plt.plot(num_epochs, val_loss, label='test_perplexity', c='green')
    plt.title('Convergence Graph- {}'.format(suffix))
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.savefig('./plots/Convergence Graph- {}.png'.format(suffix))


train_data, valid_data, test_data, V, idx_to_word = load_data()

ppe_train = []
ppe_val = []
seed = 12
num_steps = 35
batch_size =20
skip_step = 1
embedding_dim = 50
hidden_units = 5 #200 small model
dropout = 0.5 if USE_DROPOUT else 0.0
nb_epoch = 1 # 39
#check_data_gen(train_data, V)
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, V, skip_step=skip_step)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, V, skip_step=skip_step)
test_data_generator = KerasBatchGenerator(test_data, num_steps, batch_size, V, skip_step=skip_step)







my_init = RandomUniform(minval=-0.05, maxval=0.05, seed=seed)

model = Sequential()
model.add(Embedding(V, embedding_dim, batch_input_shape=(batch_size,num_steps) , input_length=num_steps))
if USE_GRU:
    model.add(GRU(units=hidden_units, batch_input_shape=(batch_size, num_steps, embedding_dim), dropout=dropout,
                   kernel_constraint=max_norm(5.), recurrent_constraint=max_norm(5.), bias_constraint=max_norm(5.),
                   kernel_initializer=my_init, recurrent_initializer=my_init, bias_initializer='zeros',
                   stateful=True, return_sequences=True))
    model.add(GRU(units=hidden_units, batch_input_shape=(batch_size, num_steps, embedding_dim), dropout=dropout,
                   kernel_constraint=max_norm(5.), recurrent_constraint=max_norm(5.), bias_constraint=max_norm(5.),
                   kernel_initializer=my_init, recurrent_initializer=my_init, bias_initializer='zeros',
                   stateful=True, return_sequences=False))
else:
    model.add(LSTM(units = hidden_units, batch_input_shape=(batch_size,num_steps, embedding_dim), dropout = dropout,
                   kernel_constraint=max_norm(5.), recurrent_constraint=max_norm(5.), bias_constraint=max_norm(5.),
                   kernel_initializer=my_init, recurrent_initializer=my_init, bias_initializer='zeros',
                   stateful=True, return_sequences=True))
    model.add(LSTM(units = hidden_units, batch_input_shape=(batch_size,num_steps, embedding_dim), dropout = dropout,
                   kernel_constraint=max_norm(5.), recurrent_constraint=max_norm(5.), bias_constraint=max_norm(5.),
                   kernel_initializer=my_init, recurrent_initializer=my_init, bias_initializer='zeros',
                   stateful=True, return_sequences=False))
model.add(Dropout(dropout)) #third and final drop out. If there are L layers of LSTM, there are L+1 Dropouts
model.add(Dense(units = V, activation='softmax')) # Output layer with softmax, to classify V words
print(model.summary())

for i in range(nb_epoch):
    # in the paper they decrease lr by factor of 1.15 from epoch 14, until then the lr is 1
    base_lr = 1
    decay = 1.2 if i>=6 else 1
    lr = base_lr/decay
    optimizer = SGD(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,  metrics=['categorical_accuracy'])
    #categorical accuracy, example in 3 class category prediction:
    # y_ture = [[1,0,0],[0,1,0]] y_pred = [[0.5,0.3,0.2], [0.8,0.2,0.2]]
    # then we have one correct pred and one incorrect. resulting in categorical acc= 0.5

    history = model.fit_generator(train_data_generator.generate(),
                        steps_per_epoch= len(train_data)//(batch_size*num_steps),
                        epochs =1,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps))
    perplexity_train = np.exp(history.history['loss'][0]) # PPE = exp (categorical_crossentropy)
    perplexity_val = np.exp(history.history['val_loss'][0])
    print('After Epoch : {}, PPE train : {}, PPE val : {}'.format(i+1, perplexity_train, perplexity_val))
    ppe_train.append(perplexity_train)
    ppe_val.append(perplexity_val)
    model.reset_states()

print('evaluating on Test set')
test_results = model.evaluate_generator(test_data_generator.generate(), steps= len(test_data)//(batch_size*num_steps))
print('Perplexity on Test Set: {}'.format(np.exp(test_results[0])))

model.save(os.path.join(MODEL_PATH,"model_use_gru:{}_use_dropout:{}.hdf5".format(USE_GRU, USE_DROPOUT)))

exit()

model = load_model(os.path.join(MODEL_PATH,"final_model.hdf5"))
dummy_iters = 40

example_test_generator = KerasBatchGenerator(test_data, num_steps, 1, V, skip_step=1)
print("Test data:")
num_predict = 10
true_print_out = "Actual words: "
pred_print_out = "Predicted words: "
for _ in range(num_predict):
    x_,  y_ = next(example_test_generator.generate())
    y_hat = model.predict(x_)
    predict_word = np.argmax(y_hat)
    true_word = np.argmax(y_)
    true_print_out += idx_to_word[y_] + " "
    pred_print_out += idx_to_word[predict_word] + " "
print(true_print_out)
print(pred_print_out)



