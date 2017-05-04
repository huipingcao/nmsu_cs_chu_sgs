#PBS -l nodes=1:ppn=1
#PBS -l walltime=24000:00:00

iteration=200
DATE=20161120
# solver: 0 = Normal LDA
# solver: 1 = Sparse LDA
# solver: 8 = Alias LDA
# solver: 9 = F+LDA - word-by-word
# solver: 10 = F+LDA - doc-by-doc
# solver: 100 = subsampling Normal LDA
for Znum in 1000 #
do
    for f in nips #enron nips kos nytimes pubmed 
    do
        for m in 108
        do
            for ratio in 0.2 0.4 0.6 0.8 1.0
            do
                sampler_id=/data/chu/lda-data/nomad_lda_data/output/$f-$m-Z$Znum-ratio$ratio-${DATE}
                ../splda $Znum $iteration /data/chu/lda-data/nomad_lda_data/$f $m $sampler_id $ratio 
                sleep 10
                for iratio in 0.2 0.4 0.6 0.8 1.0
                do
                    for t in 208
                    do
                        sampler_id=/data/chu/lda-data/nomad_lda_data/output/$f-$t-Z$Znum-ratio$ratio-iratio$iratio-${DATE}
                        ../setcover_subsampling $Znum $iteration /data/chu/lda-data/nomad_lda_data/$f $t $sampler_id $ratio $iratio
                    done
                done
            done
        done
        for m in 8
        do
            sampler_id=/data/chu/lda-data/nomad_lda_data/output/$f-$m-Z$Znum-${DATE}
            ../splda $Znum $iteration /data/chu/lda-data/nomad_lda_data/$f $m $sampler_id
        done
        wait
        sleep 10
    done
done

for Znum in 2000 5000 8000 10000 #
do
    for f in nips #enron nips kos nytimes pubmed 
    do
        for m in 108
        do
            for ratio in 1.0
            do
                sampler_id=/data/chu/lda-data/nomad_lda_data/output/$f-$m-Z$Znum-ratio$ratio-${DATE}
                ../splda $Znum $iteration /data/chu/lda-data/nomad_lda_data/$f $m $sampler_id $ratio 
                sleep 10
                for iratio in 0.3
                do
                    for t in 208
                    do
                        sampler_id=/data/chu/lda-data/nomad_lda_data/output/$f-$t-Z$Znum-ratio$ratio-iratio$iratio-${DATE}
                        ../setcover_subsampling $Znum $iteration /data/chu/lda-data/nomad_lda_data/$f $t $sampler_id $ratio $iratio
                    done
                done
            done
        done
        for m in 8
        do
            sampler_id=/data/chu/lda-data/nomad_lda_data/output/$f-$m-Z$Znum-${DATE}
            ../splda $Znum $iteration /data/chu/lda-data/nomad_lda_data/$f $m $sampler_id
        done
        wait
        sleep 10
    done
done
