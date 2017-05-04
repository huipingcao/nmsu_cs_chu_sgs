//#define DEBUG_SUBSAMPLE_LDA
#ifdef DEBUG_SUBSAMPLE_LDA
#define POSITION printf("file: %s function: %s line: %d\n", __FILE__, __FUNCTION__, __LINE__);
#endif

#define DEBUG_COMPRESSION

#include <cmath>

// subsampling : Normal sampling doc-by-doc
void run_lda_subsampling(lda_smat_t &training, lda_smat_t &test, char* sampler_id, int k, int iters, double alpha, double beta, double ratio){ //{{{
    typedef std::vector<unsigned> vec_t;
    typedef std::vector<vec_t> mat_t;

    vec_t Nt(k, 0);
    mat_t Nwt(training.nr_words, vec_t(k));
    mat_t Ndt(training.nr_docs, vec_t(k));
    training.initialize_Z_docwise(k);

    for(auto d = 0U; d <training.nr_docs; d++) {
        for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
            auto w = training.word_idx[idx];
            for(auto zidx = training.Z_ptr[idx]; zidx != training.Z_ptr[idx+1]; zidx++) {
                auto t = training.Zval[zidx];
                Nt[t]++;
                Ndt[d][t]++;
                Nwt[w][t]++;
            }
        }
    }
    for(int iter = 1; iter <= 1; iter++) {
        for(auto d = 0U; d < training.nr_docs; d++) {
            if(training.doc_ptr[d] == training.doc_ptr[d+1])
                continue;
            auto &Nd = Ndt[d];

            for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
                auto w = training.word_idx[idx];
                auto &Nw = Nwt[w];

                auto idx_s = training.Z_ptr[idx], idx_e = training.Z_ptr[idx+1]-1; //word start and end index (inclusive) in training.Z_ptr
                auto total_count = idx_e - idx_s + 1;
                size_t group_count = (size_t)ceil(total_count*ratio); //sample group_count words as a group (assign the same topic to them)
                group_count = group_count==0?1:group_count; //total_count
                size_t sampled_count = 0;
                auto group_s = idx_s, group_e = idx_s+group_count-1;//group start and end (includsive) index 

                while (sampled_count < total_count) { //sample one group together
                    for(auto z_it=group_s; z_it<=group_e; z_it++) {
                        size_t old_topic = training.Zval[z_it];
                        --Nt[old_topic]; --Nw[old_topic];--Nd[old_topic];
                    }
                    unsigned seed = clock();
                    size_t new_topic = rand_r(&seed) % k;

                    for(auto z_it=group_s; z_it<=group_e; z_it++) 
                        training.Zval[z_it] = new_topic;
                    Nt[new_topic] += group_count; Nw[new_topic] += group_count; Nd[new_topic] += group_count;

                    sampled_count += (group_e-group_s+1);
                    group_s = group_e+1; 
                    if (group_count+sampled_count>total_count || group_count==0) {
                        group_count = idx_e-group_s+1; //treat all of what remains as one group
                    }
                    group_e = group_s+group_count-1;
                }
            }
        }
    }

    double alphabar = alpha*k;
    double betabar = beta*training.nr_words;
    auto compute_training_LL = [&]()->double { // {{{
        double LL = 0;
        for(auto doc = 0U; doc < training.nr_docs; doc++) {
            auto &Nd = Ndt[doc];
            size_t sum_Nd = 0;
            for(auto t = 0; t < k; t++) if(Nd[t]){
                LL += lgamma(alpha+Nd[t])-lgamma(alpha);
                sum_Nd += Nd[t];
            }
            LL += lgamma(alphabar) - lgamma(alphabar+sum_Nd);
        }
        size_t nonZeroTypeTopics = 0;
        for(auto word = 0U; word < training.nr_words; word++) {
            auto &Nw = Nwt[word];
            for(auto t = 0; t < k; t++) if(Nw[t]){
                nonZeroTypeTopics++;
                LL += lgamma(beta+Nw[t]);
            }
        }
        size_t valid_topics = 0;
        for(auto t = 0; t < k; t++) if(Nt[t] != 0) {
            LL -= lgamma(betabar+Nt[t]);
            valid_topics++;
        }
        LL += valid_topics*lgamma(betabar)-nonZeroTypeTopics*lgamma(beta);
        return LL;
    }; // }}}

    std::vector<double> params(k);
    auto sampler = [&](double sum)->int { // {{{
        double urnd = drand48()*sum; int t = 0;
        while(1) { //TODO unroll loop
            urnd -= params[t];
            if(urnd < 0) break;
            t++;
        }
        return t;
    }; // }}}

    FILE *log_file; 
    char *cp = new char[1000];
    strcpy(cp, sampler_id);
    strcat(cp, ".log.txt");
    log_file = fopen(cp, "w+");

    fprintf(log_file, "init LL %.6g\n", compute_training_LL());
    size_t total_cputime = 0;

    for(int iter = 1; iter <= iters; iter++) {
        std::clock_t start_time = clock();
        size_t corpus_total_count=0, corpus_sampled_count=0;
        for(auto d = 0U; d < training.nr_docs; d++) {
            auto &Nd = Ndt[d];
            for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
                auto w = training.word_idx[idx];
                auto &Nw = Nwt[w];

                auto idx_s = training.Z_ptr[idx], idx_e = training.Z_ptr[idx+1]-1; //word start and end index (inclusive) in training.Z_ptr
                auto total_count = idx_e - idx_s + 1;
                size_t group_count = (size_t)ceil(total_count*ratio); //sample group_count words as a group (assign the same topic to them)
                group_count = group_count==0?1:group_count; //total_count
                size_t sampled_count = 0;
                auto group_s = idx_s, group_e = idx_s+group_count-1;//group start and end (includsive) index 

                while (sampled_count<total_count) { //sample one group together
                    size_t old_z = training.Zval[group_s];
                    Nw[old_z] -= group_count; Nd[old_z] -= group_count; Nt[old_z] -= group_count;//decreament count

                    double sum = 0;
                    for(auto t = 0; t < k; t++) { //TODO unroll loop
                        params[t] = (alpha+Nd[t])*(beta+Nw[t])/(betabar+Nt[t]);
                        sum += params[t];
                    }
                    size_t new_z = sampler(sum);
                    training.Zval[group_s] = new_z;

                    Nw[new_z] += group_count; Nd[new_z] += group_count; Nt[new_z] += group_count;//increament count
                    sampled_count += (group_e-group_s+1);
                    group_s = group_e+1; 
                    if (group_count==0 || group_count+sampled_count>total_count) {
                        group_count = idx_e-group_s+1; //treat all of what remains as one group
                    }
                    group_e = group_s+group_count-1;
#ifdef DEBUG_COMPRESSION
                    corpus_sampled_count++;
#endif
                }
#ifdef DEBUG_COMPRESSION
                corpus_total_count += total_count;
#endif
            }
        }
#ifdef DEBUG_COMPRESSION
        if (iter==1)
            fprintf(log_file, "corpus_total_count=%lu, corpus_sampled_count=%lu , ratio=%.2f\n", corpus_total_count, corpus_sampled_count, ratio);
#endif
        size_t singleiter_time = std::clock()-start_time;
        total_cputime += singleiter_time;
        fprintf(log_file, "iter %d LL %.6g cputime %.2f iter-time %.2f\n", iter, compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000), (double)singleiter_time/(CLOCKS_PER_SEC/1000));
        fflush(log_file);
    }

    print_parameters(Nt,Nwt,Ndt, sampler_id);


    //TODO comment later
    for(int iter = 1; iter <= 1; iter++) {
        std::clock_t start_time = clock();
        size_t corpus_total_count=0, corpus_sampled_count=0;
        for(auto d = 0U; d < training.nr_docs; d++) {
            auto &Nd = Ndt[d];
            for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
                auto w = training.word_idx[idx];
                auto &Nw = Nwt[w];

                auto idx_s = training.Z_ptr[idx], idx_e = training.Z_ptr[idx+1]-1; //word start and end index (inclusive) in training.Z_ptr
                auto total_count = idx_e - idx_s + 1;
                size_t group_count = (size_t)ceil(total_count*ratio); //sample group_count words as a group (assign the same topic to them)
                group_count = group_count==0?1:group_count; //total_count
                size_t sampled_count = 0;
                auto group_s = idx_s, group_e = idx_s+group_count-1;//group start and end (includsive) index 

                while (sampled_count<total_count) { //sample one group together
                    size_t old_z = training.Zval[group_s];
                    for (auto i=group_s; i<=group_e; i++)
                        training.Zval[i] = old_z;

                    sampled_count += (group_e-group_s+1);
                    group_s = group_e+1; 
                    if (group_count==0 || group_count+sampled_count>total_count) {
                        group_count = idx_e-group_s+1; //treat all of what remains as one group
                    }
                    group_e = group_s+group_count-1;
                }
            }
        }
    }
    print_doc_word_topic_triple(training, sampler_id); 
    //TODO comment later

    calculate_epsilon(training, test, log_file, Nt, Nwt, Ndt, k, iters, alpha, beta, ratio);
    fclose(log_file);
} // }}}
