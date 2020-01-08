#!/bin/bash
#SBATCH -p normal
#SBATCH -N 4
#SBATCH -J pre_scrach
#SBATCH -o pre_cscd_r_128_from_scrach.out
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4

DIR=`pwd`
hostfile=${DIR}/tmp
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}

for i in `cat ${hostfile}`
do
    echo ${i} slots=4
done > ${DIR}/hostfile-tmp

num_node=`cat ${hostfile} | uniq | wc -l`
((num_DCU=${num_node}*4))

export MIOPEN_USER_DB_PATH=/tmp/tensorflow-miopen-${USER}-2.8
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export HOROVOD_HIERARCHICAL_ALLREDUCE=1

mpirun -np ${num_DCU} --hostfile ${DIR}/hostfile-tmp -mca pml ucx -x UCX_TLS=sm,rc,rocm_cpy,rocm_gdr,rocm_ipc -x LD_LIBRARY_PATH -mca coll_hcoll_enable 0 --bind-to none \
    python run_pretraining_hvd.py\
 --input_file /public/home/zzx6320/lh/Projects/bert/data/cscibert_pre_training/pre_training_R_cscd_128.tfrecord\
 --bert_config_file /public/home/zzx6320/lh/Projects/bert/models/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12/bert_config.json\
 --output_dir output/Pre1_cscd_R_128_64_from_scrach\
 --max_seq_length 128\
 --do_train True\
 --do_eval True \
 --train_batch_size 64\
 --learning_rate 2e-5\
 --num_train_steps 500000\
 --save_checkpoints_steps 1000\