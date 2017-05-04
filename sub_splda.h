void run_splda_subsampling(lda_smat_t &training, lda_smat_t &test, char* sampler_id, int k, int iters, double alpha, double beta, double ratio) { // {{{
    // initialize count vectors
    typedef std::vector<spidx_t<unsigned>> spmat_t;	
    std::vector<size_t> Nt(k, 0);
    spmat_t Nwt(training.nr_words, spidx_t<unsigned>(k));
    spmat_t Ndt(training.nr_docs, spidx_t<unsigned>(k));
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
    size_t corpus_total_count=0, corpus_sampled_count=0;
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
                corpus_total_count += total_count;
                size_t group_count = (size_t)ceil(total_count*ratio); //sample group_count words as a group (assign the same topic to them)
                group_count = group_count==0?1:group_count; //total_count
                size_t sampled_count = 0;
                auto group_s = idx_s, group_e = idx_s+group_count-1;//group start and end (includsive) index 

                while (sampled_count < total_count) { //sample one group together
                    corpus_sampled_count++;
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
    for(auto &Nw: Nwt) Nw.gen_nz_idx();
    for(auto &Nd: Ndt) Nd.gen_nz_idx();

    double alphabar = alpha*k;
    double betabar = beta*training.nr_words;

    auto compute_training_LL = [&]()->double { // {{{
        double LL = 0;
        for(auto doc = 0U; doc < training.nr_docs; doc++) {
            auto &Nd = Ndt[doc];
            size_t sum_Nd = 0;
            for(auto t : Nd.nz_idx) {
                LL += lgamma(alpha+Nd[t])-lgamma(alpha);
                sum_Nd += Nd[t];
            }
            LL += lgamma(alphabar) - lgamma(alphabar+sum_Nd);
        }
        size_t nonZeroTypeTopics = 0;
        for(auto word = 0U; word < training.nr_words; word++) {
            auto &Nw = Nwt[word];
            for(auto t : Nw.nz_idx) {
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

    FILE *log_file; 
    char *cp = new char[1000];
    strcpy(cp, sampler_id);
    strcat(cp, ".log.txt");
    log_file = fopen(cp, "w+");
    fprintf(log_file, "init LL %.6g\n", compute_training_LL());

    std::vector<double> Cstatic(k);
    std::vector<entry_t<double> > C(k);
    size_t total_cputime = 0;

    for(int iter = 1; iter <= iters; iter++) {
        size_t nr_A=0, nr_B=0, nr_C=0;
        std::clock_t start_time = clock();
        for(auto d = 0U; d < training.nr_docs; d++) {
            if(training.doc_ptr[d] == training.doc_ptr[d+1])
                continue;
            // per-document variables
            double Asum=0, Bsum=0, Csum=0;
            auto &Nd = Ndt[d];
            for(auto t = 0; t < k; t++) {
                double reg_denom = Nt[t]+betabar;
                size_t reg_ndt = Nd[t];
                //Asum += beta*alpha/reg_denom;
                Asum += 1.0/reg_denom;
                if(reg_ndt==0) 
                    Cstatic[t] = alpha/reg_denom;
                else {
                    Cstatic[t] = (reg_ndt+alpha)/reg_denom;
                    //Bsum += beta*reg_ndt/reg_denom;
                    Bsum += reg_ndt/reg_denom;
                }
            }
            Asum *= beta*alpha;
            Bsum *= beta;

            for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
                auto w = training.word_idx[idx];
                auto &Nw = Nwt[w];

                auto idx_s = training.Z_ptr[idx], idx_e = training.Z_ptr[idx+1]-1; //word start and end index (inclusive) in training.Z_ptr
                auto total_count = idx_e - idx_s + 1;
                size_t group_count = ceil(total_count*ratio); //sample group_count words as a group (assign the same topic to them)
                size_t sampled_count = 0;
                auto group_s = idx_s, group_e = idx_s+group_count-1;//group start and end (includsive) index 

                while (sampled_count < total_count) { //sample one group together
                    size_t old_topic = training.Zval[group_s]; 
                    double reg_denom = Nt[old_topic]+betabar;
                    size_t reg_ndt = Nd[old_topic];
                    Nd[old_topic] -= group_count;

                    // removal old_topic
                    Asum -= beta*alpha/reg_denom;
                    if(reg_ndt)
                        Bsum -= beta*reg_ndt/reg_denom;
                    reg_ndt -= group_count;
                    reg_denom -= group_count;
                    Nt[old_topic] -= group_count;
                    Asum += beta*alpha/reg_denom;
                    if(reg_ndt==0) {
                        Nd.pop(old_topic);
                        Cstatic[old_topic] = alpha/reg_denom;
                    } else {
                        Bsum += beta*reg_ndt/reg_denom;
                        Cstatic[old_topic] = (reg_ndt+alpha)/reg_denom;
                    }
                    // Csum requires re-computation everytime.
                    Csum = 0.0;
                    size_t nz_C = 0;
                    size_t* ptr_old_topic = NULL;
                    for(auto &t: Nw.nz_idx) {
                        if(t == old_topic) {
                            Nw[t] -= group_count;
                            if(Nw[t]==0) 
                                ptr_old_topic = &t;
                        }
                        if(Nw[t]>0) {
                            C[nz_C].idx = t;
                            C[nz_C].value = Cstatic[t]*Nw[t];
                            Csum += C[nz_C].value;
                            nz_C++;
                        }
                    }
                    if(ptr_old_topic) 
                        Nw.remove_ptr(ptr_old_topic);

                    int new_topic=-1;
                    double sample = drand48()*(Asum+Bsum+Csum); 
                    if(sample < Csum) { // sampling new topic
                        auto *ptr = C.data();
                        while( (sample -= ptr->value) > 0)
                            ptr++;
                        new_topic = ptr->idx;
                        nr_C++;
                    } else {
                        sample -= Csum;
                        if(sample < Bsum) {
                            sample /= beta;
                            for(auto &t : Nd.nz_idx) {
                                sample -= Nd[t]/(Nt[t]+betabar);
                                if(sample <= 0) {
                                    new_topic = t;
                                    break;
                                }
                            }
                            nr_B++;
                        } else {
                            sample -= Bsum;
                            sample /= (alpha*beta);
                            for(auto t = 0; t < k; t++) {
                                sample -= 1.0/(betabar+Nt[t]);
                                if(sample <= 0) {
                                    new_topic = t;
                                    break;
                                }
                            }
                            nr_A++;
                        }
                    } // end sampling new topic

                    training.Zval[group_s] = new_topic;

                    // Add counts for the new_topic
                    reg_denom = Nt[new_topic]+betabar;
                    reg_ndt = Nd[new_topic];
                    Nd[new_topic] += group_count;

                    Asum -= beta*alpha/reg_denom;
                    if(reg_ndt)
                        Bsum -= beta*reg_ndt/reg_denom;
                    reg_ndt += group_count;
                    reg_denom += group_count;
                    Nt[new_topic] += group_count;
                    Nw[new_topic] += group_count;
                    if(reg_ndt==group_count)
                        Nd.push(new_topic);
                    if(Nw[new_topic]==group_count)
                        Nw.push(new_topic);
                    Asum += beta*alpha/reg_denom;
                    Bsum += beta*reg_ndt/reg_denom;
                    Cstatic[new_topic] = (reg_ndt+alpha)/reg_denom;

                    sampled_count += group_count;
                    group_s = group_e+1; 
                    if (group_count+sampled_count>total_count || group_count==0) {
                        group_count = idx_e-group_s+1; //treat all of what remains as one group
                        //group_count=1;  //treat each of what remains as one group
                    }
                    group_e = group_s+group_count-1;
                }
            }
        }
        if (iter==1)
            fprintf(log_file, "corpus_total_count=%lu, corpus_sampled_count=%lu , ratio=%.2f\n", corpus_total_count, corpus_sampled_count, ratio);

        size_t singleiter_time = (std::clock()-start_time);
        total_cputime += singleiter_time;
        double all = (double)(nr_A+nr_B+nr_C);
        fprintf(log_file, "iter %d nr_A %.2f nr_B %.2f nz_C %.2f LL %.6g cputime %.2f iter-time %.2f\n", iter, (double)nr_A/all, (double)nr_B/all, (double)nr_C/all, compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000), (double)singleiter_time/(CLOCKS_PER_SEC/1000));
        fflush(log_file);
    }
    print_parameters(Nt,Nwt,Ndt,sampler_id);
    calculate_epsilon(training, test, log_file, Nt, Nwt, Ndt, k, iters, alpha, beta, ratio);
    fprintf(log_file, "push_call_count=%d, remove_call_count=%d\n", push_call_count, remove_call_count);
    fclose(log_file);
} // }}}
