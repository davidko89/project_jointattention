# project_jointattention
* author: Chanyoung ko
* date: 2022-03-31
---
## Objective
Create a classification model for ASD(Autism Spectrum Disorder) and TD(Typical development)

## Project structure
checkpoint
code
data
    proc_data
        cnn_ija
        cnn_rja_high
        cnn_rja_low
        proc_ija
        proc_rja_high
        proc_rja_low
    raw_data
        ija
        rja_high
        rja_low
     
## Data
### [`data/dataset_videos.csv`](data/dataset_videos.csv)
* file_name
* task
    1. IJA
    2. RJA_low
    3. RJA_high_BL
    4. RJA_high_BR
    5. RJA_high_Lt
    6. RJA_high_Rt
* label: 0 - TD, 1 - ASD
