#PBS -l nodes=1:ppn=20
#PBS -l walltime=24000:00:00

iteration=300
DATE=20160208
# solver: 0 = Normal LDA
# solver: 1 = Sparse LDA
# solver: 8 = Alias LDA
# solver: 9 = F+LDA - word-by-word
# solver: 10 = F+LDA - doc-by-doc
# solver: 100 = subsampling: Normal LDA

for Znum in 10
do
    for dratio in 1.01 #1.02 1.03 1.04 1.05 1.06 1.07 1.08 1.09 1.1
    do
        Dnum=1000
        f=D$Dnum-Z$Znum-r$dratio
        for m in 100 101 108 110
        do
            for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
            do
                ./splda $Znum $iteration /data/chu/lda-data/syn_data/$f $m $ratio > syn-output/$f-$m-Z$Znum-ratio$ratio-${DATE}.txt &
            done
        done
        for m in 0 1 8 9 10
        do
            ./splda $Znum $iteration /data/chu/lda-data/syn_data/$f $m > syn-output/$f-$m-Z$Znum-${DATE}.txt &
        done
        wait
    done
done
