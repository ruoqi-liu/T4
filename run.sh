export CUDA_VISIBLE_DEVICES=1
TAU=$1
date=$(date +"%Y-%m%d")
time=$(date +"%H%M%S")
batch_aug=true
epochs=50
batch_size=32
lr=5e-5
seed=66


# run mimic

mkdir -p checkpoints/${date}
mkdir -p log/${date}

# pretrain
if ${batch_aug}; then
echo "pretrain ps estimator..."
save_model=checkpoints/${date}/${time}_${TAU}_pretrain.pt
log_file=log/${date}/${time}_${TAU}_pretrain.log
python pre_train.py --epochs ${epochs} --batch_size ${batch_size} --learning_rate ${lr} --pre_window ${TAU} --save_model ${save_model} --log_file ${log_file} --seed ${seed}
fi

echo "train main model..."
outdir="results_mimic/${date}"
mkdir -p ${outdir}
for ratio in 0.4; do

for seed in {101..150}; do

pretrained_model=checkpoints/${date}/${time}_${TAU}_pretrain.pt
save_model=checkpoints/${date}/${time}_${TAU}_${ratio}_${seed}.pt
log_file=log/${date}/${time}_${TAU}_${ratio}_${seed}.log
output_mimic=${outdir}/${time}_${TAU}_${ratio}
output_amsterdamdb=${outdir}/${time}_${TAU}_${ratio}

param="--epochs ${epochs} --batch_size ${batch_size} --learning_rate ${lr} --save_model ${save_model} --pretrained_model ${pretrained_model} --log_file ${log_file}"
param="${param} --aug_ratio ${ratio} --pre_window ${TAU} --output_mimic ${output_mimic} --output_amsterdamdb ${output_amsterdamdb} --seed ${seed}"
python main.py ${param}

done
done