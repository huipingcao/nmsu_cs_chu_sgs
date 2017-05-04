void run_alias_lda_subsampling(lda_smat_t &training, lda_smat_t &test, char* sampler_id, int k, int iters, double alpha, double beta, double ratio){ //{{{
    typedef std::vector<spidx_t<unsigned>> spmat_t;
    typedef std::vector<unsigned> vec_t;
    typedef std::vector<vec_t> mat_t;
    vec_t Nt(k, 0);
    mat_t Nwt(training.nr_words, vec_t(k));
    spmat_t Ndt(training.nr_docs, spidx_t<unsigned>(k));
    std::vector<sample_pool_t> sample_pools(training.nr_words, sample_pool_t(k));
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
    for(auto &Nd: Ndt) Nd.gen_nz_idx();

    for(auto &pool : sample_pools) 
        pool.reserve(k);
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
        while(1) {
            urnd -= params[t];
            if(urnd < 0) break;
            t++;
        }
        return t;
    }; // }}}

    std::vector<unsigned> pdw_idx(k);
    std::vector<double> pdw(k), pdw_real(k), qwqw(k);
    alias_table pdw_sampler(k), A(k);
    alias_table qw_sampler(k);

    FILE *log_file; 
    char *cp = new char[1000];
    strcpy(cp, sampler_id);
    strcat(cp, ".log.txt");
    log_file = fopen(cp, "w+");

    fprintf(log_file, "init LL %.6g\n", compute_training_LL());

    size_t nr_MH_steps = 2;
    size_t total_cputime = 0;
    for(int iter = 1; iter <= iters; iter++) {
        size_t nr_p = 0, nr_q = 0;
        size_t total_MH_steps = 0, rejected_MH_steps = 0;
        std::clock_t start_time = clock();
        for(auto d = 0U; d < training.nr_docs; d++) {
            auto &Nd = Ndt[d];
            for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
                auto w = training.word_idx[idx];
                auto &Nw = Nwt[w];
                auto &pool = sample_pools[w];

                auto idx_s = training.Z_ptr[idx], idx_e = training.Z_ptr[idx+1]-1; //word start and end index (inclusive) in training.Z_ptr
                auto total_count = idx_e - idx_s + 1;
                size_t group_count = (size_t)ceil(total_count*ratio); //sample group_count words as a group (assign the same topic to them)
                group_count = group_count==0?1:group_count; //total_count
                size_t sampled_count = 0;
                auto group_s = idx_s, group_e = idx_s+group_count-1;//group start and end (includsive) index 

                while (sampled_count < total_count) { //sample one group together
                    size_t old_topic = training.Zval[group_s];
                    Nw[old_topic] -= group_count;
                    Nd[old_topic] -= group_count;
                    Nt[old_topic] -= group_count;
                    size_t new_topic = -1;

                    if(1) {
                        // construct Pdw, pdw, pdw_idx {{{
                        double Pdw = 0.0;
                        size_t nz_p = 0;
                        size_t* ptr_old_topic = NULL;
                        for(auto &t: Nd.nz_idx) {
                            if(Nd[t]>0) {
                                pdw_idx[nz_p] = t;
                                pdw[nz_p] = Nd[t]*(Nw[t]+beta)/(Nt[t]+betabar);
                                Pdw += pdw[nz_p];
                                pdw_real[t] = pdw[nz_p];
                                nz_p++;
                            } else 
                                ptr_old_topic = &t;
                        }
                        if(ptr_old_topic) 
                            Nd.remove_ptr(ptr_old_topic);
                        auto get_pdw_real = [&](size_t t)->double { // {{{
                            auto begin = Nd.nz_idx.begin();
                            auto end = Nd.nz_idx.end();
                            auto it = std::lower_bound(begin,end, t);
                            if(it != end) 
                                return pdw[it-begin];
                            else 
                                return 0.0;
                        }; // }}}
                        //}}}
                        pdw_sampler.build_table(nz_p, pdw.data());

                        // Perform MH steps {{{
                        auto &qw = pool.qw;
                        double &Qw = pool.Qw;
                        int s = old_topic;
                        double p_s = (Nw[s]+beta)*(Nd[s]+alpha)/(Nt[s]+betabar);; 
                        total_MH_steps += nr_MH_steps;
                        for(auto step = 1; step <= nr_MH_steps; step++) {
                            if(pool.empty()) {
                                qw.resize(k);
                                for(size_t t = 0U; t < k; t++) {
                                    qw[t] = alpha*(Nw[t]+beta)/(Nt[t]+betabar);
                                }
                                pool.refill(A);
                            }
                            size_t t;
                            double rnd = drand48()*(Pdw+Qw);
                            if(rnd <= Pdw) 
                                t = pdw_idx[pdw_sampler.sample(rnd/Pdw, drand48())];
                            //t = Nd.nz_idx[pdw_sampler.sample(rnd/Pdw, drand48())];
                            else 
                                t = pool.next_sample();
                            double p_t = (Nw[t]+beta)*(Nd[t]+alpha)/(Nt[t]+betabar);
                            if(step == 0){
                                s = t;
                                p_s = (Nw[s]+beta)*(Nd[s]+alpha)/(Nt[s]+betabar);
                                continue;
                            }
                            double pi = (double)(p_t*(pdw_real[s]+qw[s]))/(double)(p_s*(pdw_real[t]+qw[t]));
                            //double pi = (double)(p_t*(get_pdw_real(s)+qw[s]))/(double)(p_s*(get_pdw_real(t)+qw[t]));
                            if(drand48() <= pi) {
                                p_s = p_t;
                                s = t;
                            } else {
                                rejected_MH_steps++;
                            }
                        } 
                        new_topic = s;
                        // }}}
                        for(auto &t: Nd.nz_idx) pdw_real[t] = 0;
                    } 
                    else if (0){  // {{{
                        double sum = 0;
                        for(auto t = 0; t < k; t++) 
                            params[t] = sum+=(alpha+Nd[t])*(beta+Nw[t])/(betabar+Nt[t]);
                        new_topic = std::upper_bound(params.begin(), params.end(), sum*drand48()) - params.begin();
                    } else if (0) {
                        auto &qw = qwqw;
                        double sum = 0;
                        for(auto t = 0; t < k; t++) {
                            qw[t] = (alpha+Nd[t])*(beta+Nw[t])/(betabar+Nt[t]);
                            sum += qw[t];
                        }
                        qw_sampler.build_table(qw);
                        new_topic = qw_sampler.sample(drand48(), drand48());
                        //new_topic = std::upper_bound(qw.begin(), qw.end(), sum*drand48()) - qw.begin();
                    } else if (0) {
                        auto &qw = qwqw;
                        double Qw = 0;
                        double Pdw = 0;
                        qw.resize(k);
                        for(size_t t = 0U; t < k; t++) {
                            qw[t] = alpha*(Nw[t]+beta)/(Nt[t]+betabar);
                            pdw[t] = Nd[t]*(Nw[t]+beta)/(Nt[t]+betabar);
                            Qw += qw[t];
                            Pdw += pdw[t];
                        }
                        qw_sampler.build_table(qw);
                        pdw_sampler.build_table(pdw);
                        double rnd = drand48()*(Pdw+Qw);
                        if(rnd <= Pdw) {
                            rnd = rnd/Pdw;
                            //new_topic = pdw_idx[pdw_sampler.sample(rnd, drand48())];
                            new_topic = pdw_sampler.sample(rnd, drand48());
                            nr_p++;
                        } else {
                            new_topic = qw_sampler.sample((rnd-Pdw)/Qw, drand48());
                            nr_q++;
                        }
                    } else if (0) {

                        // construct Pdw, pdw, pdw_idx {{{
                        double Pdw = 0.0;
                        size_t nz_p = 0;
                        size_t* ptr_old_topic = NULL;
                        for(auto &t: Nd.nz_idx) {
                            if(Nd[t]>0) {
                                pdw_idx[nz_p] = t;
                                pdw[nz_p] = Nd[t]*(Nw[t]+beta)/(Nt[t]+betabar);
                                Pdw += pdw[nz_p];
                                pdw_real[t] = pdw[nz_p];
                                nz_p++;
                            } else 
                                ptr_old_topic = &t;
                        }
                        if(ptr_old_topic) 
                            Nd.remove_ptr(ptr_old_topic);
                        //}}}
                        pdw_sampler.build_table(nz_p, pdw.data());

                        auto &qw = qwqw;
                        double Qw = 0;
                        qw.resize(k);
                        for(size_t t = 0U; t < k; t++) {
                            qw[t] = alpha*(Nw[t]+beta)/(Nt[t]+betabar);
                            Qw += qw[t];
                        }
                        qw_sampler.build_table(qw);
                        double rnd = drand48()*(Pdw+Qw);
                        if(rnd <= Pdw) {
                            new_topic = pdw_idx[pdw_sampler.sample(rnd/Pdw, drand48())];
                            nr_p++;
                        } else {
                            new_topic = qw_sampler.sample((rnd-Pdw)/Qw, drand48());
                            nr_q++;
                        }
                    } else if(0) {
                        // construct Pdw, pdw, pdw_idx {{{
                        double Pdw = 0.0;
                        size_t nz_p = 0;
                        size_t* ptr_old_topic = NULL;
                        for(auto &t: Nd.nz_idx) {
                            if(Nd[t]>0) {
                                pdw_idx[nz_p] = t;
                                pdw[nz_p] = Nd[t]*(Nw[t]+beta)/(Nt[t]+betabar);
                                Pdw += pdw[nz_p];
                                pdw_real[t] = pdw[nz_p];
                                nz_p++;
                            } else 
                                ptr_old_topic = &t;
                        }
                        if(ptr_old_topic) 
                            Nd.remove_ptr(ptr_old_topic);
                        //}}}
                        pdw_sampler.build_table(nz_p, pdw.data());

                        auto &qw = pool.qw;
                        double &Qw = pool.Qw;

                        qw.resize(k);
                        Qw = 0;
                        for(size_t t = 0U; t < k; t++) {
                            qw[t] = alpha*(Nw[t]+beta)/(Nt[t]+betabar);
                            Qw += qw[t];
                        }
                        qw_sampler.build_table(qw);

                        int s = old_topic;
                        double p_s = (Nw[s]+beta)*(Nd[s]+alpha)/(Nt[s]+betabar);; 
                        for(auto step = 0; step <= nr_MH_steps; step++) {
                            size_t t;
                            double rnd = drand48()*(Pdw+Qw);
                            if(rnd <= Pdw) 
                                t = pdw_idx[pdw_sampler.sample(rnd/Pdw, drand48())];
                            else 
                                t = qw_sampler.sample((rnd-Pdw)/Qw, drand48());
                            double p_t = (Nw[t]+beta)*(Nd[t]+alpha)/(Nt[t]+betabar);
                            if(step == 0){
                                s = t;
                                p_s = (Nw[s]+beta)*(Nd[s]+alpha)/(Nt[s]+betabar);
                                continue;
                            }
                            double pi = (double)(p_t*(pdw_real[s]+qw[s]))/(double)(p_s*(pdw_real[t]+qw[t]));
                            if(drand48() <= pi) {
                                p_s = p_t;
                                s = t;
                            } else {
                                rejected_MH_steps++;
                            }
                        }
                        for(auto &t: Nd.nz_idx)
                            pdw_real[t] = 0;
                        new_topic = s;

                    } // }}}
                    training.Zval[group_s] = new_topic;
                    Nw[new_topic] += group_count;
                    Nd[new_topic] += group_count;
                    Nt[new_topic] += group_count;
                    if(Nd[new_topic]==group_count) Nd.push(new_topic);

                    sampled_count += (group_e-group_s+1);
                    group_s = group_e+1; 
                    if (group_count+sampled_count>total_count || group_count==0) {
                        group_count = idx_e-group_s+1; //treat all of what remains as one group
                    }
                    group_e = group_s+group_count-1;
                }
            }
        }
        size_t singleiter_time = std::clock()-start_time;
        total_cputime += singleiter_time;
        double ac_rate = (double) (total_MH_steps-rejected_MH_steps)/(total_MH_steps);
        fprintf(log_file, "iter %d LL %.6g cputime %.2f iter-time %.2f ac_rate %.2f\n", iter, compute_training_LL(), 
                (double)total_cputime/(CLOCKS_PER_SEC/1000), 
                (double)singleiter_time/(CLOCKS_PER_SEC/1000), 
                ac_rate);
        fflush(log_file);
    }
    print_parameters(Nt,Nwt,Ndt,sampler_id);
    calculate_epsilon(training, test, log_file, Nt, Nwt, Ndt, k, iters, alpha, beta, ratio);
    fprintf(log_file, "push_call_count=%d, remove_call_count=%d\n", push_call_count, remove_call_count);
    fclose(log_file);
} // }}}

