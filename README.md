# project_jointattention
* author: Chanyoung ko
* date: 2022-03-31
---
## Objective
Create a classification model for ASD(Autism Spectrum Disorder) and TD(Typical development)

## Project structure
* checkpoint
* code
* data
* proc_data
    1. cnn_ija
    2. cnn_rja_high
    3. cnn_rja_low
    4. proc_ija
    5. proc_rja_high
    6. proc_rja_low
* raw_data
    1. ija
    2. rja_high
    3. rja_low
     
## Data
### [`data/ija_videofile_with_dx.csv`](data/dataset_videos.csv)
### [`data/rja_high_videofile_with_dx.csv`](data/dataset_videos.csv)
### [`data/rja_low_videofile_with_dx.csv`](data/dataset_videos.csv)
* file_name
* task
    1. IJA
    2. RJA_low
    3. RJA_high_BL
    4. RJA_high_BR
    5. RJA_high_Lt
    6. RJA_high_Rt
* label: 0 - TD, 1 - ASD
