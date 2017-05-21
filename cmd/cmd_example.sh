#PBS -l nodes=1:ppn=1
#PBS -l walltime=24000:00:00

iteration=100
DATE=20170520
DATA_SET=nytimes #nips nytimes pubmed 

# solver: m=0  Normal LDA
# solver: m=1  Sparse LDA
# solver: m=8  Alias LDA
# solver: m=9  F+LDA - word-by-word
for Znum in 10 # change number of topics here
do
    for m in 0 1 8 9 # loop through all comparison method
    do

        # output log and parameters are writtent to this path.
        sampler_id=/data/chu/lda-data/nomad_lda_data/output/$DATA_SET-$m-Z$Znum-${DATE}
        # IMPORTANT: change the data path to the local data path on your machine.
        ../splda $Znum $iteration /data/chu/lda-data/nomad_lda_data/$DATA_SET $m $sampler_id
    done
done
