#PBS -l nodes=1:ppn=1
#PBS -l walltime=24000:00:00

for data in nips enron nytimes pubmed
do
    ../setCover_lda 10 2 /data/chu/lda-data/nomad_lda_data/$data/ -1 0.5 0.5
done
