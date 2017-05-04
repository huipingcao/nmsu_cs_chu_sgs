#PBS -l nodes=1:ppn=48
#PBS -l walltime=24000:00:00

# test to find the best # of parallel process
DATE=20161117
# solver: 0 = Normal LDA
# solver: 1 = Sparse LDA
# solver: 8 = Alias LDA
# solver: 9 = F+LDA - word-by-word
# solver: 10 = F+LDA - doc-by-doc
# solver: 100 = subsampling Normal LDA
# for Znum in 1000 5000 10000 20000
for Znum in 100
do
    for f in nytimes #enron nips kos nytimes pubmed 
    do
        iteration=100
        ############################
        for m in 101
        do
            for ratio in 0.1 0.2 0.3 0.4 
            do
                sampler_id=/data/chu/lda-data/nomad_lda_data/output/parallel-test4-$f-$m-Z$Znum-ratio$ratio-${DATE}
                ../splda $Znum $iteration /data/chu/lda-data/nomad_lda_data/$f $m $sampler_id $ratio &
                sleep 10
            done
            wait

            for ratio in 0.5 0.6 0.7 0.8 
            do
                sampler_id=/data/chu/lda-data/nomad_lda_data/output/parallel-test4-$f-$m-Z$Znum-ratio$ratio-${DATE}
                ../splda $Znum $iteration /data/chu/lda-data/nomad_lda_data/$f $m $sampler_id $ratio &
                sleep 10
            done
            wait

            for ratio in 0.9 1.0
            do
                sampler_id=/data/chu/lda-data/nomad_lda_data/output/parallel-test4-$f-$m-Z$Znum-ratio$ratio-${DATE}
                ../splda $Znum $iteration /data/chu/lda-data/nomad_lda_data/$f $m $sampler_id $ratio &
                sleep 10
            done

            for m in 1 
            do
                sampler_id=/data/chu/lda-data/nomad_lda_data/output/parallel-test4-$f-$m-Z$Znum-${DATE}
                ../splda $Znum $iteration /data/chu/lda-data/nomad_lda_data/$f $m $sampler_id
            done
            wait
            sleep 10
        done
    done
done
