#PBS -l nodes=1:ppn=48
#PBS -l walltime=24000:00:00

sh nytimes_subsampling_comparison_splda.sh &
sh nytimes_subsampling_comparison_alias.sh &
sh nytimes_subsampling_comparison_flda.sh &
sh nytimes_subsampling_comparison_lda.sh
wait

