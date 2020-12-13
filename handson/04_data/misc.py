import tensorflow as tf


def csv_reader_dataset(file_paths, repeat=1, batch_size=32, shuffle_buffer_size=10000,
                       num_readers=5, num_read_threads=None, num_parse_threads=5):
    dataset = tf.data.Dataset.list_files(file_paths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=num_readers,
        num_parallel_calls=num_read_threads
    )
    dataset = dataset.shuffle(shuffle_buffer_size)\
        .map(preprocess, num_parallel_calls=num_parse_threads)\
        .batch(batch_size)\
        .prefetch(1)
    return dataset


@tf.function
def preprocess(line):
    defs = [0.] * num_columns + [tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - x_mean) / x_std, y


optimizer = keras.optimizers.Nadam(lr=0.01)
loss_fn = keras.losses.mean_squared_error

@tf.function
def train(model, num_epochs, batch_size=32, shuffle_buffer_size=10000,
          num_readers=5, num_read_threads=None, num_parse_threads=5):
    train_set = csv_reader_dataset(
        file_paths=train_filepaths, repeat=num_epochs, batch_size=batch_size,shuffle_buffer_size=shuffle_buffer_size,
        num_readers=num_readers, num_read_threads=num_read_threads, num_parse_threads=num_parse_threads)
    for x_batch, y_batch in train_set:
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
            loss = tf.add_n([main_loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

train(model, 5)