# %% Imports
import os
import shutil
import pickle
import datetime
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from transformer import Transformer
from lib.scheduler import LRScheduler
import utils as utils

# load config
config_file = "test_config.yaml"

# %%

def setup_tf(config_file: str):
    config_file = os.path.join("src/config_files", config_file)
    config = utils.load_config(config_file)

    # create a unique directory for the model
    model_name = config["model_name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    cwd = os.getcwd()
    model_dir = f"{cwd}/models/{model_name}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)

    # copy config file for documentation
    shutil.copy(config_file, f"{model_dir}/config.yaml")

    # set paths for tf logs
    log_dir_name = config["data"]["logs_dir_name"]
    log_dir = os.path.join(model_dir, log_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    
    #tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    tf.config.run_functions_eagerly(True)
    tf.data.experimental.enable_debug_mode()

    return config, log_dir, model_dir


def load_data(config):
    data_dir = config["data"]["data_dir"]
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    # Load data and data specific parameters
    train_test_val_ratio = config["data"]["train_test_ratio"]
    X, Y = utils.get_spring_mass_damper_data(data_dir)
    x_train, x_test, x_val, y_train, y_test, y_val = utils.split_data(X, Y, train_test_val_ratio)

    y_train_shifted = utils.shift(y_train) # decoder input
    y_test_shifted = utils.shift(y_test)
    y_val_shifted = utils.shift(y_val)


    # ensure returned values are TF tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

    # print info to console
    print(f"\nLoaded data from {data_dir}")
    print(f"Encoder input shape: {x_train.shape}")
    print(f"Decoder input and output shape: {y_train.shape}")
    print("Shapes are in order (N_sim, N_timesteps, N_dof) and (N_batch, N_timesteps, d_model) internally.\n")

    return x_train, y_train_shifted, y_train, x_test, y_test_shifted, y_test, x_val, y_val_shifted, y_val


def train_model(config, optimizer, model_dir):
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]

    checkpoint_dir = os.path.join(model_dir, config["data"]["checkpoints_dir_name"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    #data_path_weights_filename = os.path.join(model_dir, checkpoint_dir, "model_weights")

    # setup callbacks - tensorboard, checkpoints, early stopping
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq='epoch',
    )

    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    #manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=3)

    model_ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
            #data_path_weights_filename,
            filepath=checkpoint_dir, 
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    
    # works well, but slows down training
    # csv_logger = tf.keras.callbacks.CSVLogger(f"{model_dir}/training.log", append=True)

    if config["training"]["early_stopping"]:
        early_stopping_cb = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                verbose=1,
                restore_best_weights=True
            )

        callbacks = [model_ckpt_cb, tensorboard_cb, early_stopping_cb] # , csv_logger]
    else:
        callbacks = [model_ckpt_cb, tensorboard_cb]# , csv_logger]


    # train model
    start_time = datetime.datetime.now()
    history = model.fit(x=[x_train, y_train_shifted],
                        y=y_train,
                        validation_data=([x_val, y_val_shifted], y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        )

    
    with open(f'{model_dir}/trainHistoryDict', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    end_time = datetime.datetime.now()
    utils.print_timedelta(start_time, end_time)
    return model, history


def load_checkpoint(config, untrained_model, model_dir):
    checkpoint_dir = os.path.join(model_dir, config["data"]["checkpoints_dir"])
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]

    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print("Restoring from", latest_checkpoint)

    #model = tf.train.checkpoint.restore(checkpoint_dir)
    return tf.keras.models.load_model(latest_checkpoint)


def eval_model(config, model, x_test, y_test_shifted, y_test, batch_size):
    # load best weights
    model.load_weights(f"{model_dir}/checkpoints")
    results = model.evaluate([x_test, y_test_shifted], y_test, batch_size=batch_size)
    
    print("trained model - test loss (mse)", results)

    save_dir = os.path.join(model_dir, "saved_model")
    os.makedirs(save_dir, exist_ok=True)
    tf.saved_model.save(model, save_dir)


#----
config, log_dir, model_dir = setup_tf(config_file)
x_train, y_train_shifted, y_train, x_test, y_test_shifted, y_test, x_val, y_val_shifted, y_val = load_data(config)

n_seq, encoder_seq_length, d_input = x_train.shape
n_seq, decoder_seq_length, d_output = y_train.shape

model, optimizer = utils.build_transformer(config, encoder_seq_length, decoder_seq_length, d_output)

model, history = train_model(config, optimizer, model_dir)

utils.plot_train_and_val_loss(history, model_dir)



# ----

# # %% Evaluate model
print("Evaluating model on test data...")
batch_size = config["training"]["batch_size"]
results = model.evaluate([x_test, y_test_shifted], y_test, batch_size=batch_size)
print(results)

# %%
