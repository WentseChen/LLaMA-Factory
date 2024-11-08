# conda env 
source /home/scratch/wentsec/anaconda3/bin/activate 
conda activate babyai
export VLLM_WORKER_MULTIPROC_METHOD=spawn 

# hyper-parameter
log_id=0
alpha=2.5
temperature=0.2
total_iter=10
batch_size=1024
num_workers=8
env_name="BabyAI-GoToDoor-v0"
model_path="/dev/shm/fine_tuned_models/fft"
data_path="/zfsauton2/home/wentsec/LLaMA-Factory/data"
finetuned_model="/dev/shm/fine_tuned_models/fft/checkpoint"
mini_batch_size=$((batch_size/num_workers))

# set save path
rm -r $model_path
mkdir $model_path
rm -r $data_path/logs
mkdir $data_path/logs
rm $data_path/phase*

# rollout
cd /zfsauton2/home/wentsec/LLaMA-Factory
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli api examples/inference/llama3_vllm.yaml \
>> $data_path/logs/result.txt &
SERVER_PID=$!
cd /zfsauton2/home/wentsec/LLaMA-Factory/collect
for j in $(seq 0 7);
do
    CUDA_VISIBLE_DEVICES=$j python rollout_cot.py \
    --env_name $env_name --rank $j \
    --batch_size $mini_batch_size --file_path $data_path \
    --temperature $temperature \
    >> $data_path/logs/result$((j))_0.txt &
done
wait $(jobs -p | grep -v $SERVER_PID)
sleep 10.0
cd /zfsauton2/home/wentsec/LLaMA-Factory/collect
python merge.py --file_path $data_path --batch_size $batch_size --phase 3
kill $SERVER_PID
while true; do
    if ! ps -ef | grep -v grep | grep wentse | grep python > /dev/null; then
        ps -ef | grep wentse | grep collect_traj | awk '{print $2}' | xargs kill
        sleep 5.0
        break
    else
        ps -ef | grep wentse | grep ray | awk '{print $2}' | xargs kill
        ps -ef | grep wentse | grep python | awk '{print $2}' | xargs kill -9
        sleep 0.1
    fi
done

rm $data_path/babyai.json
cd /zfsauton2/home/wentsec/LLaMA-Factory/collect
python generate2.py --file_path $data_path --env_name $env_name --batch_size $batch_size

# train
cd /zfsauton2/home/wentsec/LLaMA-Factory
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/llama3_base.yaml &
old_file="/dev/shm/fine_tuned_models/fft/checkpoint-20/trainer_state.json"
while true; do
    if [ -f $old_file ]; then
        sleep 0.1
        kill $TRAINING_PID
        ps -ef | grep wentse | grep python | awk '{print $2}' | xargs kill -9
        ps -ef | grep wentse | grep wandb | awk '{print $2}' | xargs kill
        sleep 0.1
        break
    else
        sleep 0.1
    fi
done


# for i in $(seq 0 $((total_iter-1)));
# do

#     rm $data_path/phase*

#     # rename the old model
#     if [ $i -gt 0 ]; then
#         old_folder=$finetuned_model-"$((i*20))"
#         rm -r $finetuned_model
#         mv $old_folder $finetuned_model
#     fi

#     # rollout
#     cd /zfsauton2/home/wentsec/LLaMA-Factory/collect
#     for j in $(seq 0 7);
#     do
#         if [ $i -eq 0 ]; then
#             CUDA_VISIBLE_DEVICES=$j python rollout.py \
#             --env_name $env_name --rank $j \
#             --batch_size $mini_batch_size --file_path $data_path \
#             --temperature $temperature \
#             >> $data_path/logs/result$((log_id))_0.txt &
#         else
#             CUDA_VISIBLE_DEVICES=$j python rollout.py \
#             --env_name $env_name --rank $j \
#             --batch_size $mini_batch_size --file_path $data_path \
#             --temperature $temperature \
#             --model_name $finetuned_model \
#             >> $data_path/logs/result$((log_id))_$((i)).txt &
#         fi
#     done
#     wait
#     python merge.py --file_path $data_path --batch_size $batch_size --phase 1

#     echo "Finish rollout ====================================="

#     # reflection
#     cd /zfsauton2/home/wentsec/LLaMA-Factory
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli api examples/inference/llama3_vllm.yaml &
#     SERVER_PID=$!
#     cd /zfsauton2/home/wentsec/LLaMA-Factory/collect
#     python reflect.py --file_path $data_path --batch_size $batch_size --mini_batch_size 128 # 8:128, 4:64
#     kill $SERVER_PID
#     while true; do
#         if ! ps -ef | grep -v grep | grep wentse | grep python > /dev/null; then
#             ps -ef | grep wentse | grep collect_traj | awk '{print $2}' | xargs kill
#             sleep 5.0
#             break
#         else
#             ps -ef | grep wentse | grep ray | awk '{print $2}' | xargs kill
#             ps -ef | grep wentse | grep python | awk '{print $2}' | xargs kill -9
#             sleep 0.1
#         fi
#     done

#     echo "Finish reflection ====================================="

#     # in-context learning
#     cd /zfsauton2/home/wentsec/LLaMA-Factory/collect
#     for j in $(seq 0 7);
#     do
#         if [ $i -eq 0 ]; then
#             CUDA_VISIBLE_DEVICES=$j python icl.py \
#             --env_name $env_name --rank $j \
#             --file_path $data_path --temperature $temperature &
#         else
#             CUDA_VISIBLE_DEVICES=$j python icl.py \
#             --env_name $env_name --rank $j --file_path $data_path \
#             --temperature $temperature --model_name $finetuned_model &
#         fi
#     done
#     wait
#     python merge.py --file_path $data_path --batch_size $batch_size --phase 3

#     # generate dataset
#     rm $data_path/babyai.json
#     cd /zfsauton2/home/wentsec/LLaMA-Factory/collect
#     python generate.py --file_path $data_path --env_name $env_name --alpha $alpha

#     # train
#     cd /zfsauton2/home/wentsec/LLaMA-Factory
#     if [ $i -eq 0 ]; then
#         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/llama3_base.yaml &
#     else
#         CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_full/llama3_finetuned.yaml &
#     fi
#     old_file="/dev/shm/fine_tuned_models/fft/checkpoint-"$((i*20+20))"/trainer_state.json"
#     while true; do
#         if [ -f $old_file ]; then
#             sleep 0.1
#             kill $TRAINING_PID
#             ps -ef | grep wentse | grep python | awk '{print $2}' | xargs kill -9
#             ps -ef | grep wentse | grep wandb | awk '{print $2}' | xargs kill
#             sleep 0.1
#             break
#         else
#             sleep 0.1
#         fi
#     done

# done


