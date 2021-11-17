import os

from tensorflow import keras

from dataset import VideoDataset


IMG_SIZE = 299
BATCH_SIZE = 1

MAX_SEQ_LENGTH = 32
NUM_FEATURES = 2048

labels = ['backhand_one_hand', 'backhand_two_hand', 'forehand_one_hand', 'forehand_two_hand', 'ready', 'serve', 'volley_backhand', 'volley_forehand']
idx2label = {i: name for i, name in enumerate(labels)}


def get_video_paths(base):
    ret = []
    for path, _, files in os.walk(base):
        for file in files:
            ret.append(os.path.join(path, file))
    return ret


def build_feature_extractor():
    feature_extractor = keras.applications.Xception(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = keras.applications.xception.preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def get_sequence_model():
    class_vocab = len(idx2label)

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    
    x = keras.layers.Bidirectional(keras.layers.GRU(256, return_sequences=True))(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.Bidirectional(keras.layers.GRU(128))(x)    
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    output = keras.layers.Dense(class_vocab, activation="softmax")(x)


    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]
    )
    return rnn_model

# Utility for running experiments.
def inference_video(test, filepath="./models/curing/curing_bi_gru"):
    feature_extractor = build_feature_extractor()
    test_dataset = VideoDataset(test, feature_extractor, BATCH_SIZE, MAX_SEQ_LENGTH, NUM_FEATURES, IMG_SIZE)
    seq_model = get_sequence_model()
    seq_model.load_weights(filepath)
    result = seq_model.predict(
        test_dataset[0],
    )
    return idx2label[result.argmax(-1)[0]]

