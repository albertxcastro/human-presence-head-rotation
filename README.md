# CNN-Based Human Presence Detection via Head Rotation Analysis

This repo is composed of two main models.

## Regressor model
The regressor model is a `CNN`-based model trained using a sub-dataset derived from the 300W-LP dataset, explicitly focusing on the head position attributes.
The original dataset used is the 300W-LP dataset. It contains various features, but only the head position attributes (yaw, pitch, roll) were extracted for this project.

![Regressor training model drawio](https://github.com/user-attachments/assets/cdadfed0-77e5-4cad-9a02-40e027b510b9)


The regressor model is trained using only yaw angle values. The model architecture is shown in the figure below.

![CNN-Regressor architecture drawio](https://github.com/user-attachments/assets/18e06ec1-dff2-4472-92b3-1618868d1d26)

The input layer accepts a `128x128` pixel image as input and returns a predicted yaw angle value. For more details, please look at the [regressor](notebooks/regressor.ipynb) notebook.

## Yaw time-series generator
This notebook contains the necessary scripts written in Python to generate time-series data from a video clip. The times series contains yaw angles detected in the video clip across the time, 
and the angles are predicted using the regressor model described in the section above.

The steps to generate the times series are shown in the figure below.

![yaw-dataseries-generation drawio](https://github.com/user-attachments/assets/72b5bda1-afd7-4f37-a160-067b0384d54c)

The steps are:
1. Users record a clip of themselves. The video clip must contain their head, and they must move their head from one side to the other.
2. The video clip is split into frames. The frame rate is five per second (`1` frame taken every `0.2` seconds).
3. The frames are fed into the regressor model, which generates a dictionary, where the `key` is the time the frame was taken and the `value` is the predicted yaw angle from the input image.

Example output taken from a `~5.2` seconds video clip:
```python
{
    0.2: 2.5284867,
    0.4: 2.749594,
    0.6000000000000001: 2.1379437,
    0.8: 1.7823929,
    1.0: 1.6482912,
    1.2000000000000002: 5.0864453,
    1.4000000000000001: 14.992984,
    1.6: 16.431183,
    1.8: 16.039757,
    2.0: 16.063456,
    2.2: 12.145712,
    2.4000000000000004: 1.3173454,
    2.6: -1.07616,
    2.8000000000000003: -0.4528833,
    3.0: 0.51127344,
    3.2: 13.553554,
    3.4000000000000004: 19.458012,
    3.6: 18.421278,
    3.8000000000000003: 22.45929,
    4.0: 22.07545,
    4.2: 22.521818,
    4.4: 23.279245,
    4.6000000000000005: 21.392345,
    4.800000000000001: 19.092186,
    5.0: 14.48045,
    5.2: -4.107289
}
```

The data represents the following pattern:

![download](https://github.com/user-attachments/assets/b8c5a742-0cf2-4071-81db-5cfee4d6024c)

For more details, please have a look at the [yaw-time-series-generator](notebooks/yaw-time-series-generator.ipynb) notebook.

## Classifier model
The classifier model is a `RandomForestClassifier`. The training dataset was generated manually and then expanded using synthetic samples from real samples. 
The dataset contains yaw angle time series data as described in the previous section, labeled as `0` for instances without head rotation and `1` when there is a head rotation pattern.

Before classifying an unseen instance, feature engineering is performed to extract the following features from the time series data:
* __Peaks and troughs in the data__: This helps us check that there's movement in the video clip.
* __The number of peaks.__
* __The number of troughs.__
* __The amplitude between peaks and troughs__: This helps us detect whether there were head rotation movements. Large amplitude indicates a significant rotation movement, while small amplitude
  values mean a small rotation variation.
* __Average yaw value__: This indicates the average direction the head is facing over time.
* __Yaw range__: The peak-to-peak range helps us capture the difference between the maximum and minimum yaw values, helping to detect the total range of motion.
* __Yaw standard deviation__: This helps us measure the variability of the head motion. A high standard deviation indicates that the head movement is highly variable, while a low standard
  deviation means a lower variable motion.

After the features are extracted from the times series data, they are fed into the classifier to classify the unseen instance, expecting either `0` if there are no rotation movements, or `1` if there
is a correct rotation head pattern.

![classifier drawio](https://github.com/user-attachments/assets/e8639436-fef2-4eb9-8aae-a4e9a1ee6919)

The classifier aims to detect fake impersonations in video clips: People using photos or digital images of a person.

For more details, please see the [yaw-time-series-generator](notebooks/classifier.ipynb) notebook.
