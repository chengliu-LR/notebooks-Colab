import numpy as np
import pandas as pd
landmarks_frame = pd.read_csv('./faces/face_landmarks.csv')
print(landmarks_frame)
n = 1
img_name = landmarks_frame.iloc[n, 0]
print('image name: {}'.format(img_name))
landmarks = landmarks_frame.iloc[n, 1:]
print(type(landmarks), landmarks, '\n')
#landmarks = np.array(landmarks)

# following will also call np.array()
landmarks = landmarks.to_numpy()
print(type(landmarks), landmarks.shape)