#!/bin/bash

base_path=~/experiments/supplementary
suffix=supplementary/frames

datasets=(
    2-metronomes-slo-mo
    better-bg
    face-2-attributes
    metronome-slo-mo
    yuhe
    transformer
)
names=(
    two-metronomes
    face-expressions-1
    face-expressions-3
    metronome
    face-expressions-2
    transformer
)

for (( i=0; i<${#datasets[*]}; ++i))
do
    dataset=${datasets[$i]}
    name=${names[$i]}

    mkdir -p videos/real/${name}/
    if [[ "${dataset}" == "transformer" ]]
    then
        ours_path=${base_path}/ours-${dataset}/${suffix}
    else
        ours_path=ours-${dataset}
    fi

    python combine_videos.py \
        ${ours_path} \
        frames/${dataset} \
        videos/real/${name}/ours.mp4 \
        --text texts/ours.jpg 

    python combine_videos.py \
        ${base_path}/hypernerf_ds-${dataset}/${suffix} \
        frames/${dataset} \
        videos/real/${name}/hypernerf_ds.mp4 \
        --text texts/hypernerf_ds.jpg

    python combine_videos.py \
        ${base_path}/hypernerf_ds_proj-${dataset}/${suffix} \
        frames/${dataset} \
        videos/real/${name}/hypernerf_ds_proj.mp4 \
        --text texts/hypernerf_ds_proj.jpg

    python combine_videos.py \
        ${ours_path} \
        ${base_path}/hypernerf_ds_proj-${dataset}/${suffix} \
        ${base_path}/hypernerf_ds-${dataset}/${suffix} \
        frames/${dataset} \
        videos/real/${name}/combined.mp4 \
        --text texts/ours.jpg texts/hypernerf_ds_proj.jpg texts/hypernerf_ds.jpg
done



dataset=processed-trio-train-color-fixed
name=synthetic
python combine_videos.py \
    ${base_path}/ours-${dataset}-enc_attr_mask/${suffix} \
    frames/${dataset} \
    videos/real/${name}/ours.mp4 \
    --text texts/ours.jpg \
    --width 256 \
    --height 256


python combine_videos.py \
    ${base_path}/hypernerf_ds-${dataset}/${suffix} \
    frames/${dataset} \
    videos/real/${name}/hypernerf_ds.mp4 \
    --text texts/hypernerf_ds.jpg \
    --width 256 \
    --height 256

python combine_videos.py \
    ${base_path}/hypernerf_ds_proj-${dataset}/${suffix} \
    frames/${dataset} \
    videos/real/${name}/hypernerf_ds_proj.mp4 \
    --text texts/hypernerf_ds_proj.jpg \
    --width 256 \
    --height 256

python combine_videos.py \
    ${base_path}/ours-${dataset}-enc_attr_mask/${suffix} \
    ${base_path}/hypernerf_ds_proj-${dataset}/${suffix} \
    ${base_path}/hypernerf_ds-${dataset}/${suffix} \
    frames/${dataset} \
    videos/real/${name}/combined.mp4 \
    --text texts/ours.jpg texts/hypernerf_ds_proj.jpg texts/hypernerf_ds.jpg \
    --width 256 \
    --height 256