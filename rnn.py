import tensorflow as tf
import numpy as np
import os
import time


def rnn(clustering, seq_length, k):
    """
    Perform RNN on clustering 
    """
    dataset = preprocess(clustering, seq_length)
     # Batch size
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Build the model
    # Length of the vocabulary in chars = k
    vocab_size = k
    # The embedding dimension
    embedding_dim = 256
    # Number of RNN units
    rnn_units = 1024
    model = build_model(
                vocab_size = vocab_size,
                embedding_dim=embedding_dim,
                rnn_units=rnn_units,
                batch_size=BATCH_SIZE)

    EPOCHS=10
    train_model(dataset, model, EPOCHS, seq_length)
    
  
def predict(model, clustering, seq_length):
    """
    Performs prediction with the given model
    """
    model.reset_states()
    prediction_L = []
    
    for i in range(len(clustering)-seq_length):
        input = clustering[i:i+seq_length]
        input = tf.expand_dims(input, 0)
        predictions = model(input)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        prediction_L.append(predicted_id)
    print(len(prediction_L))
    
    return prediction_L


def preprocess(clustering, seq_length):
    """
    Preprocess data for building the model
    """
    # define length of each sequence
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(clustering)
    # Put clustering in sequences with length of 5
    sequences = dataset.batch(seq_length+1, drop_remainder=True)

    # Using sliding window method
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
    return sequences.map(split_input_target)

   
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """
    Build the rnn model, called in rnn() and main
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def train_model(dataset, model, epochs, seq_length):
    """
    Train the rnn model, called in rnn()
    """
    model.compile(optimizer='adam', loss=loss)

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints_' + str(seq_length)
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)
    
    # Actual training process
    history = model.fit(dataset, epochs=epochs, callbacks=[checkpoint_callback])


def loss(labels, logits):
    """
    Calculate loss
    """
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


def accuracy(predictions_L, actual_L):
    """
    Caclulate accuracy of the rnn model, called in main
    """
    correct = 0
    for i in range(len(actual_L)):
        if (predictions_L[i]==actual_L[i]):
            correct += 1
    return correct/len(actual_L)


