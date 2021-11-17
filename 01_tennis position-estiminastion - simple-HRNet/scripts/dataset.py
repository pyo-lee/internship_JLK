import cv2
import math
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import Sequence


SEED = 41
PATIENCE = 4
WEIGHTS = None

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


class VideoDataset(Sequence):
    def __init__(self, video_paths, feature_extractor, batch_size, max_seq_length, num_features, image_shape):
        self.video_paths = video_paths
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
        self.IMG_SIZE = image_shape
        self.MAX_SEQ_LENGTH = max_seq_length
        self.NUM_FEATURES = num_features

        labels = ['backhand_one_hand', 'backhand_two_hand', 'forehand_one_hand', 'forehand_two_hand', 'ready', 'serve', 'volley_backhand', 'volley_forehand']
        self.label_table = {name: i for i, name in enumerate(labels)}


    def __len__(self):
        return math.ceil(len(self.video_paths) / self.batch_size)


    def __getitem__(self, idx):
        batch_x = self.video_paths[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_data = self.prepare_all_videos(batch_x)
        return batch_data


    def load_video(self, path, max_frames=0, resize=(299, 299)):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames)
    

    def prepare_all_videos(self, video_paths):
        num_samples = len(video_paths)
        # labels = np.array([self.label_table[os.path.basename(os.path.dirname(path))] for path in video_paths])
        
        frame_masks = np.zeros(shape=(num_samples, self.MAX_SEQ_LENGTH), dtype="bool")
        frame_features = np.zeros(
            shape=(num_samples, self.MAX_SEQ_LENGTH, self.NUM_FEATURES), dtype="float32"
        )

        # For each video.
        for idx, path in enumerate(video_paths):
            # Gather all its frames and add a batch dimension.
            frames = self.load_video(path)
            frames = frames[None, ...]

            # Initialize placeholders to store the masks and features of the current video.
            temp_frame_mask = np.zeros(shape=(1, self.MAX_SEQ_LENGTH,), dtype="bool")
            temp_frame_featutes = np.zeros(
                shape=(1, self.MAX_SEQ_LENGTH, self.NUM_FEATURES), dtype="float32"
            )

            # Extract features from the frames of the current video.
            for i, batch in enumerate(frames):
                video_length = batch.shape[0]
                length = min(self.MAX_SEQ_LENGTH, video_length)
                for j in range(length):
                    temp_frame_featutes[i, j, :] = self.feature_extractor.predict(
                        batch[None, j, :]
                    )
                temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

            frame_features[idx,] = temp_frame_featutes.squeeze()
            frame_masks[idx,] = temp_frame_mask.squeeze()

        return (frame_features, frame_masks)