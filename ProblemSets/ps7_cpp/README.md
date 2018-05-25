# Problem Set 7: Motion History Images

## Problem 1: Frame-differenced MHI

### Part a
Binary sequences computed at each frame of the `PS7A1P1T1.avi` video. These are masks where if motion is greater than a given threshold, the pixel's value is set to 1; otherwise it is set to 0.

Frame 10 | Frame 20 | Frame 30
---|---|---
<img src="./output/ps7-1-a-1.png" width="320"/> | <img src="./output/ps7-1-a-2.png" width="320"/> | <img src="./output/ps7-1-a-3.png" width="320"/>


### Part b
Motion history images computed from the binary frames. See [the config file](../../config/ps7.yaml) for the values of tau and the "last frame" of each sequence.

Action 1 | Action 2 | Action 3
---|---|---
<img src="./output/ps7-1-b-1.png" width="320"/> | <img src="./output/ps7-1-b-2.png" width="320"/> | <img src="./output/ps7-1-b-3.png" width="320"/>


## Problem 2: Recognition using MHIs

### Part a
Confusion matrices for the central and scale-invariant moments. The image moments are used as features for a KNN classifier and the action number (1/2/3) is used as the label. For each input, that feature and label is removed from the dataset and tested against the remaining data in the set.

Mu | Eta
---|---
<img src="./output/ps7-2-a-1.png" width="320"/> | <img src="./output/ps7-2-a-2.png" width="320"/>

### Part b
"Non-cheating" way, where each person is removed from the training data and is instead used as testing data.

Person 1 | Person 2 | Person 3 | Average of all
---|---|---|---
<img src="./output/ps7-2-b-1.png" width="320"/> | <img src="./output/ps7-2-b-2.png" width="320"/> | <img src="./output/ps7-2-b-3.png" width="320"/> | <img src="./output/ps7-2-b-4.png" width="320"/>