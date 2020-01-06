#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -J test_cla_multi_gpus

export MIOPEN_USER_DB_PATH=/tmp/tensorflow-miopen-${USER}-2.8
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export HOROVOD_HIERARCHICAL_ALLREDUCE=1

nodefile=$(generate_pbs_nodefile)
np=$(cat $nodefile|sort|uniq |wc -l)

np=$(($np*4))

mpirun -np $np --allow-run-as-root -hostfile $nodefile -mca pml ucx -x UCX_TLS=sm,rc,rocm_cpy,rocm_gdr,rocm_ipc -x LD_LIBRARY_PATH -mca coll_hcoll_enable 0 --bind-to none \
    python run_classifier_hvd.py \
    --task_name=cla \
    --do_lower_case=true \
    --do_train=true \
    --do_eval=true \
    --data_dir=/public/home/zzx6320/lh/Projects/bert/data/data_wanfang_16000 \
    --vocab_file=/public/home/zzx6320/lh/Projects/bert/models/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/vocab.txt \
    --bert_config_file=/public/home/zzx6320/lh/Projects/bert/models/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/bert_config.json \
    --init_checkpoint=/public/home/zzx6320/lh/Projects/bert/models/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/bert_model.ckpt \
    --max_seq_length=400 \
    --train_batch_size=8 \
    --learning_rate=2e-5 \
    --num_train_epochs=2.0 \
    --output_dir=output/test_horovod_cla_16000 \
    --cla_nums=16