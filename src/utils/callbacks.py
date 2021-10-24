import tensorflow as tf
import os
import numpy as np
import time

def get_timestamp(name):
    timestamp = time.asctime().replace(" ","_").replace(":","_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name


def get_callbacks(config,x_train_small):
    logs = config['logs']
    unique_dir_name = get_timestamp("tb_logs")
    TENSORBOARD_ROOT_LOG_DIR = os.path.join(logs["logs_dir"],logs["TENSORBOARD_ROOT_LOG_DIR"],unique_dir_name)

    os.makedirs(TENSORBOARD_ROOT_LOG_DIR,exist_ok=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir= TENSORBOARD_ROOT_LOG_DIR)

    file_writer = tf.summary.create_file_writer(logdir=TENSORBOARD_ROOT_LOG_DIR)

    with file_writer.as_default():
        images = np.reshape(x_train_small[10:30],(20,28,28,1))    ###<<< 20,28,28,1
        tf.summary.image("20 handwritten digit sample", images,max_outputs=25, step=0)

    params = config["params"]
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience = params["patience"], restore_best_weights=params["restore_best_weights"])
    
    artifacts = config["artifacts"]
    ckpt_dir = os.path.join(artifacts["artifacts_dir"],artifacts["CHECKPOINT_DIR"])
    os.makedirs(ckpt_dir,exist_ok=True)

    Ckpt_path = os.path.join(ckpt_dir, "model_ckpt.h5")

    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(Ckpt_path , save_best_only=True)

    return [tensorboard_cb,early_stopping_cb,checkpointing_cb]