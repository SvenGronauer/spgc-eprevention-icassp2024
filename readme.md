
# SPGC ICASSP 2024 Challenge

## Installation
Create a new conda environment:

```bash
conda create -n spgc python=3.9
conda activate spgc
```

Install the requirements:

```bash
pip install -r requirements.txt
```

#### Data
Features are already pre-processed for you and saved into the directory: `data/track_1_features`. 



### Run Training (on Track 1)

You are on default branch for track 2. However, you can train the ensemble model also on track 1 via:
```bash
python train.py --num_patients 9 --window_size 24 --features_path data/track_1_new_features/ --dataset_path data/track_1/ --cores 8 --save_path /home/YOUR-NAME/track2/
```
Note: `--cores X` sets the number of CPU cores used for data loading. 

### Run Training (on Track 2)
You are on default branch for track 1. However, you can train also on track 2 via:

```
python train.py --cores 8 --features_path data/track_2_new_features/ --dataset_path data/track_2/ --save_path /home/YOUR-NAME/track1/
```

## Approach
Changes made to the baseline:

**Pre-Processing**

 - Instead of ignoring parts of the day where data is missing, impute with the feature's median for *the specific 5-minute segment of the day* in question, for that patient
 - Add step data

**Model**

 - Instead of training the model to predict the user ID, change the objective to predicting the time stamps (minute of the day) of the sequence, based on  the input features. The idea behind this is that a relapse may lead to changes in a patient's daily routine, which will then be reflected in the model's prediction error.
 - Change `dim_feedforward_encoder` from `2048` to `64`
 - Change `d_model` from `32` to `64`

**Training**

 - Train a separate model for each patient. Analogous to the baseline, check the relapse detection accuracy on the validation set after each epoch and save a model if it surpasses the best score so far. 
 - Use a sequence length of 72 for Track 1 and a sequence length of 24 for Track 2
 - Change batch size from `16` to `64`
 - Change `dropout_encoder` from `0.2`to `0.1`

**Note**: I haven't systematically tested the impact of the changes in dropout, batch size, or model dimensions, so those might not actually be doing anything 😬

**Outlier Detection**

Instead of using an Elliptic Envelope detector, collect the mean prediction error for each day in the training set and the validation set. For each day in the validation set, calculate the mean-normalized prediction error:
$$ 
e_{\text{norm}} = \frac{{e_{\text{val}} - \overline{e}_{\text{train}}}}{\text{max}(e_{\text{train}}) -\text{min}(e_{\text{train}})} 
$$

Designate any day with a normalized prediction error above 0 as a relapse day:

$$
\mathrm{score}(e_{\text{norm}}) = \begin{cases}
    0 & \text{if } e_{\text{norm}} \leq 0 \\
    1 & \text{if } e_{\text{norm}} > 0 
\end{cases}
$$


## Instructions
### Data Pre-Processing
If you have not done so already, extract the features as outlined in the baseline repository under "Data" -> "Feature extraction". Afterwards, run the following command for track 1:  
```bash  
python new_preprocessing.py --raw_dataset_path data/track_1/ --preprocessed_data_path data/track_1_features/  
```  
  
and for track 2 run:  
  
```bash  
python new_preprocessing.py --raw_dataset_path data/track_2/ --preprocessed_data_path data/track_2_features/  
```

### Training
To train models for track 1 run:
```bash  
python train.py --window_size 72 --num_patients 9 --features_path data/track_1_features --dataset_path data/track_1 --save_path checkpoints_track_1
```

To train models for track 2 run:
```bash  
python train.py --window_size 24 --num_patients 8 --features_path data/track_2_features --dataset_path data/track_2 --save_path checkpoints_track_2
```

## Results
### Metrics
Results on the validation set after applying all proposed changes (averaged over 5 runs):


| Track 1 | PR-AUC | ROC-AUC | AVG |  
|---------|--------|---------|-----|  
| Random Chance | 0.326 | 0.500 | 0.413 |  
| Baseline | 0.472 | 0.614 | 0.543 |
| *With Changes* | *0.680 ± 0.005* | *0.665 ± 0.007* | *0.672 ± 0.005* |

| Track 2 | PR-AUC | ROC-AUC | AVG |  
|---------|--------|---------|-----|  
| Random Chance | 0.349 | 0.500 | 0.424 |  
| Baseline | 0.452 | 0.594 | 0.522 |
| *With Changes* | *0.632 ± 0.008* | *0.606 ± 0.036* | *0.619 ± 0.022* |

### Visualizations

Example visualization for the model's prediction error for patient 5 in track 1. The error is shown per 5-minute segment, i.e., there are 288 (24 * 60 / 5) segments per day. Where the model makes predictions on overlapping segments, they were averaged. The plots show that, in this case, the prediction error is lower on validation set days without relapses. Many relapse days show errors especially after midnight and in the early hours of the morning, indicating a change in activity pattern. 

![Train set](img/train_5.png)

![No relapse](img/val_5_no_rel.png)

![Relapse](img/val_5_rel.png)


