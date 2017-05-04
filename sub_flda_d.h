// F+LDA Sampling with 2 terms doc-by-doc (best for now) _test
void run_splda_fast_2_test_subsampling(lda_smat_t &training, lda_smat_t &test, char* sampler_id, int k, int iters, double alpha, double beta, double ratio) { // {{{
    FILE *log_file; 
    char *cp = new char[1000];
    strcpy(cp, sampler_id);
    strcat(cp, ".log.txt");
    log_file = fopen(cp, "w+");

    // initialize count vectors
    typedef std::vector<spidx_t<unsigned>> spmat_t;
    std::vector<size_t> Nt(k, 0);
    spmat_t Nwt(training.nr_words, spidx_t<unsigned>(k));
    spmat_t Ndt(training.nr_docs, spidx_t<unsigned>(k));
    training.initialize_Z_docwise(k);
    std::vector<size_t> nnz_w(training.nr_words), nnz_d(training.nr_docs);
    size_t	nnz_t = 0;

    for(auto d = 0U; d <training.nr_docs; d++) {
        for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
            auto w = training.word_idx[idx];
            for(auto zidx = training.Z_ptr[idx]; zidx != training.Z_ptr[idx+1]; zidx++) {
                auto t = training.Zval[zidx];
                Nt[t]++;
                Ndt[d][t]++;
                Nwt[w][t]++;
                nnz_w[w]++;
                nnz_d[d]++;
                nnz_t++;
            }
        }
    }
    for(int iter = 1; iter <= 1; iter++) { // initial topic in groups 
        for(auto d = 0U; d < training.nr_docs; d++) {
            if(training.doc_ptr[d] == training.doc_ptr[d+1])
                continue;
            // per-document variables (D)
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

    for(auto &Nw: Nwt) Nw.gen_nz_idx();
    for(auto &Nd: Ndt) Nd.gen_nz_idx();

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

    fprintf(log_file, "init LL %.6g\n", compute_training_LL());

    htree_t B(k), D(k);
    //std::vector<entry_t<double> > C(k); // for inner product
    std::vector<double> C(k); // for inner product
    std::vector<size_t> C_idx(k); // for inner product
    std::vector<double>A(k,0.0); 
    //size_t threshold = (size_t) floor(2.0*k/(log2((double)k)));
    size_t threshold = (size_t) floor(15.0*k/(log2((double)k)));
    size_t total_cputime = 0;
    //size_t inner_product_time = 0, inner_sample_time = 0;
    std::clock_t tmp_start = clock();
    for(auto t = 0; t < k; t++)
        D[t] = alpha/(Nt[t]+betabar);
    D.init_dense();
    total_cputime += clock() - tmp_start;

    for(int iter = 1; iter <= iters; iter++) {
        size_t nr_A=0, nr_B=0, nr_C=0, nr_D=0;
        std::clock_t start_time = clock();
        size_t corpus_total_count=0, corpus_sampled_count=0;
        for(auto d = 0U; d < training.nr_docs; d++) {
            if(training.doc_ptr[d] == training.doc_ptr[d+1])
                continue;
            // per-document variables (D)
            auto &Nd = Ndt[d];
            if(Nd.nz_idx.size() > threshold) {
                for(auto t: Nd.nz_idx)
                    D.true_val[t] = (alpha+Nd[t])/(Nt[t]+betabar);
                D.init_dense();
            } else {
                for(auto t: Nd.nz_idx)
                    D.set_value(t, (alpha+Nd[t])/(Nt[t]+betabar));
            }
            /*
               D.true_val[t] = (alpha+Nd[t])/(Nt[t]+betabar);
               D.init_sparse(Nd.nz_idx.data(), Nd.nz_idx.size());
               */

            for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
                auto w = training.word_idx[idx];
                auto &Nw = Nwt[w];

                auto idx_s = training.Z_ptr[idx], idx_e = training.Z_ptr[idx+1]-1; //word start and end index (inclusive) in training.Z_ptr
                auto total_count = idx_e - idx_s + 1;
                corpus_total_count += total_count;
                size_t group_count = ceil(total_count*ratio); //sample group_count words as a group (assign the same topic to them)
                // group_count = group_count==0?1:group_count; //total_count
                size_t sampled_count = 0;
                auto group_s = idx_s, group_e = idx_s+group_count-1;//group start and end (includsive) index 

                while (sampled_count < total_count) { //sample one group together
                    corpus_sampled_count += 1;
                    // handle each occurrence of word w
                    // Remove counts for the old_topic
                    size_t old_topic = training.Zval[group_s]; 
                    Nt[old_topic] -= group_count;
                    double reg_denom = 1.0/(Nt[old_topic]+betabar);
                    Nw[old_topic] -= group_count; Nd[old_topic] -= group_count;
                    double D_old = D.true_val[old_topic];
                    double D_new = reg_denom*(alpha+Nd[old_topic]);
                    D.true_val[old_topic] = D_new;
                    bool updated = false;

                    // Handle Inner Product (Part C) {{{
                    size_t nz_C = 0;
                    size_t *ptr_Nw_old_topic = NULL;
                    double Csum = 0;
                    for(auto &t : Nw.nz_idx) {
                        if(Nw[t]) {
                            /*
                               C[nz_C].idx = t;
                            //C[nz_C].value = Nw[t]*(Nd[t]+alpha)*D[t];
                            //C[nz_C].value = Nw[t]*(B[t]+alpha*D[t]);
                            //C[nz_C].value = Nw[t]*(B[t]+D[t]);
                            //C[nz_C].value = Nw[t]*(B[t]);
                            //C[nz_C].value = A[t]*Nd[t];
                            C[nz_C].value = Nw[t]*D.true_val[t];
                            Csum += C[nz_C].value;
                            */
                            C[nz_C] = (Csum += Nw[t]*D.true_val[t]);
                            /*
                               C[nz_C] = Nw[t]*D.true_val[t];
                               Csum += C[nz_C];
                               */
                            C_idx[nz_C] = t;
                            nz_C++;
                        } else {
                            ptr_Nw_old_topic = &t;
                        }
                    }
                    //				if(ptr_Nw_old_topic) Nw.remove_ptr(ptr_Nw_old_topic);
                    // }}}

                    int new_topic = -1;
                    double Dsum = Csum+beta*(D.total_sum()-D_old+D_new);
                    //double Dsum = Csum+beta*D.total_sum();
                    double sample = drand48()*Dsum; 
                    //printf("sample %g Csum %g Asum %g Bsum %g Dsum %g\n", sample, Csum, Asum, Bsum, Dsum);
                    if(sample < Csum) { // {{{
                        auto *ptr = C.data();
                        new_topic = C_idx[std::upper_bound(ptr, ptr+nz_C, sample)-ptr];
                        /*
                           while((sample-=*ptr)>0)
                           ptr++;
                           new_topic = C_idx[ptr-C.data()];
                           */
                        /*
                           while((sample-=ptr->value) > 0)
                           ptr++;
                           new_topic = ptr->idx;
                           */
                        //	nr_C++;
                    } else {
                        D.update_parent(old_topic, D_new-D_old);
                        updated = true;
                        sample = (sample-Csum)/beta;
                        new_topic = D.log_sample(sample);
                        //	nr_D++;
                    } // }}}

                    training.Zval[group_s] = new_topic;

                    // Add counts for the new_topic
                    Nt[new_topic] += group_count;
                    reg_denom = 1./(Nt[new_topic]+betabar);
                    Nw[new_topic] += group_count; Nd[new_topic] += group_count;

                    if((int)old_topic != new_topic) {
                        if(Nd[old_topic]==0) Nd.pop(old_topic);
                        if(ptr_Nw_old_topic) Nw.remove_ptr(ptr_Nw_old_topic);
                        if(Nw[new_topic]==group_count) Nw.push(new_topic);
                        if(Nd[new_topic]==group_count) Nd.push(new_topic);
                        if(not updated) D.update_parent(old_topic, D_new-D_old);
                        D.set_value(new_topic, reg_denom*(alpha+Nd[new_topic]));
                    } else {
                        //if(not updated) D.update_parent(old_topic, D_new-D_old);
                        //D.set_value(new_topic, reg_denom*(beta+Nw[new_topic]));
                        if(updated) D.set_value(old_topic, D_old);
                        else D.true_val[old_topic] = D_old;
                    }
                    sampled_count += (group_e-group_s+1);
                    //group_count = (size_t) floor(group_count*ratio);// (size_t)floor(ratio*(total_count-sampled_count)); //update group count
                    group_s = group_e+1; 
                    if (group_count+sampled_count>total_count || group_count==0) {
                        group_count = idx_e-group_s+1; //treat all of what remains as one group
                        //group_count=1;  //treat each of what remains as one group
                    }
                    group_e = group_s+group_count-1;
                }
            }
            if(Nd.nz_idx.size() > threshold) {
                for(auto t: Nd.nz_idx)
                    D.true_val[t] = (alpha)/(Nt[t]+betabar);
                D.init_dense();
            } else {
                for(auto t: Nd.nz_idx)
                    D.set_value(t, (alpha)/(Nt[t]+betabar));
            }
            /*
               D.true_val[t] = alpha/(Nt[t]+betabar);
               D.init_sparse(Nd.nz_idx.data(), Nd.nz_idx.size());
               */
        }
#ifdef DEBUG_COMPRESSION
        if (iter==1)
            fprintf(log_file, "corpus_total_count=%lu, corpus_sampled_count=%lu , ratio=%.2f\n", corpus_total_count, corpus_sampled_count, ratio);
#endif
        size_t singleiter_time = (std::clock()-start_time);
        total_cputime += singleiter_time;
        double all = (double)(nr_A+nr_B+nr_C+nr_D);
        fprintf(log_file, "iter %d nr_C %.2f nr_D %.2f LL %.6g cputime %.2f iter-time %.2f\n",
                iter, (double)nr_C/all, (double)nr_D/all, 
                compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000), (double)singleiter_time/(CLOCKS_PER_SEC/1000));
        fflush(log_file);
    }

    print_parameters(Nt,Nwt,Ndt,sampler_id);
    calculate_epsilon(training, test, log_file, Nt, Nwt, Ndt, k, iters, alpha, beta, ratio);
    fprintf(log_file, "push_call_count=%d, remove_call_count=%d\n", push_call_count, remove_call_count);
    fclose(log_file);
} // }}}
