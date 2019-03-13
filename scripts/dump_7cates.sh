#! /bin/bash


train_dir=${1}
mkdir -p ${train_dir}/sn7cates
save_dir="${train_dir}/sn7cates"
for cate in "chair" "sofa" "table" "vessel" "airplane" "rifle" "car"; do
    ref_fname="${save_dir}/ref-cate-${cate}.npy"
    smp_fname="${save_dir}/smp-cate-${cate}.npy"
    python ae_dump_eval_results.py ${cate} --ref_outfname ${ref_fname} --smp_outfname ${smp_fname} ${train_dir}
done
