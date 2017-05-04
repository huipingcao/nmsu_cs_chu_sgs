#include "splda.h"
#include "SetCover.h"
#include <unordered_set>

inline bool in_set(const std::unordered_set<size_t> min_set, const size_t d) {
    return (min_set.find(d)!=min_set.end());
}

void write_min_set(std::unordered_set<size_t> min_set, const char* data_path) {
    FILE *min_set_file; 
    char *cp = new char[1000];
    strcpy(cp, data_path);
    strcat(cp, "/min_set.txt");
    min_set_file = fopen(cp, "w+");
    for(std::unordered_set<size_t>::iterator it = min_set.begin(); it!=min_set.end(); it++)
        fprintf(min_set_file, "%u\n", *it); 
    fclose(min_set_file);
}

std::unordered_set<size_t> read_min_set(const char* data_path) {
    std::unordered_set<size_t> min_set;
    FILE *min_set_file; 
    char *cp = new char[1000];
    strcpy(cp, data_path);
    strcat(cp, "/min_set.txt");
    min_set_file = fopen(cp, "r");

    size_t d = 0U;
    while(fscanf(min_set_file, "%u\n", &d)!=EOF) {
        min_set.insert(d);
    }
    fclose(min_set_file);
    return min_set;
}

inline bool exists_min_set (const char *data_path) {
    char *cp = new char[1000];
    strcpy(cp, data_path);
    strcat(cp, "/min_set.txt");

    if (FILE *file = fopen(cp, "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}
void lda_read_data(const char* srcdir, lda_smat_t &training_set, lda_smat_t &test_set, int nr_threads, smat_t::format_t fmt){//{{{
    size_t nr_docs, nr_words, nnz, Z_len;
    char filename[1024], buf[1024], suffix[12];
    FILE *fp; 
    sprintf(filename,"%s/meta",srcdir);
    //printf("filename: %s \n",filename);
    fp = fopen(filename,"r");
    if(fscanf(fp, "%lu", &nr_words) != 1) {
        fprintf(stderr, "Error: corrupted meta in line 1 of %s\n", srcdir);
        return;
    }

    //printf("nr_words: %lu \n",nr_words);

    if(fscanf(fp, "%lu %lu %lu %s", &nr_docs, &nnz, &Z_len, buf) != 4) {
        fprintf(stderr, "Error: corrupted meta in line 2 of %s\n", srcdir);
        return;
    }

    //printf("%lu %lu %lu %s\n", nr_docs, nnz, Z_len, buf);
    //printf("fmt:%lu\n",fmt);

    if(fmt == smat_t::TXT) 
        strcpy(suffix, "");
    else if(fmt == smat_t::PETSc)
        strcpy(suffix, ".petsc");
    else 
        printf("Error: fmt %d is not supported.", fmt);
    sprintf(filename,"%s/%s%s", srcdir, buf, suffix);

    //printf("filename: %s \n",filename);

    training_set.load_data(nr_docs, nr_words, nnz, filename, fmt);

    if(fscanf(fp, "%lu %lu %lu %s", &nr_docs, &nnz, &Z_len, buf) != EOF){
        sprintf(filename,"%s/%s%s", srcdir, buf, suffix);
        test_set.load_data(nr_docs, nr_words, nnz, filename, fmt);
    }
    fclose(fp);
    return ;
}

void run_lda_subsampling_iratio(lda_smat_t &training, lda_smat_t &test, const char *src, const char *sampler_id, int k, int iters, double alpha, double beta, double ratio,double iratio){ //{{{
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
                        //group_count=1;  //treat each of what remains as one group
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
        while(1) {
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
    // fclose(log_file); //TODO comments later
    // log_file = stdout;

    size_t min_set_starttime = std::clock();
    std::unordered_set<size_t> min_set;
    if (exists_min_set(src)) {
        min_set = read_min_set(src);
        fprintf(log_file, "read from min_set file size=%d, nr_docs=%d\n", min_set.size(), training.nr_docs);
    }
    else {
        std::set<size_t> tmp =  getMinimalDataSet(training);
        min_set.insert(tmp.begin(), tmp.end());
        write_min_set(min_set, src);
        fprintf(log_file, "setcover running time=%.2f, set-size=%d, total-size=%d\n", 
                (double)(std::clock()-min_set_starttime)/(CLOCKS_PER_SEC/1000), min_set.size(), training.nr_docs );
    }

    fprintf(log_file, "init LL %.6g\n", compute_training_LL());
    size_t total_cputime = 0;

    //exit(0);

    for(int iter = 1; iter <= iters; iter++) {
        std::clock_t start_time = clock();

        bool iter_uncover = (iter%((int)(iratio*10)))==0;
        size_t corpus_total_count=0, corpus_sampled_count=0;
        size_t sampled_doc = 0;
        for(int iter_counter=0; iter_counter<=1; iter_counter++) {
            std::clock_t debug_d = clock();

            if (iter_counter==1 && !iter_uncover) continue;

            for(auto d = 0U; d < training.nr_docs; d++) {
                bool inset = min_set.find(d)!=min_set.end();
                if ( iter_counter==0 && !inset)
                    continue;
                if (iter_counter==1 && inset) 
                    continue;

                sampled_doc++;
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

                    while (sampled_count<total_count) { //sample one group together
                        size_t old_z = training.Zval[group_s];
                        Nw[old_z] -= group_count; Nd[old_z] -= group_count; Nt[old_z] -= group_count;//decreament count
                        corpus_sampled_count++;

                        //-------------START draw sample for group
                        double sum = 0;
                        for(auto t = 0; t < k; t++) {
                            params[t] = (alpha+Nd[t])*(beta+Nw[t])/(betabar+Nt[t]);
                            sum += params[t];
                        }
                        size_t new_z = sampler(sum);
                        //-------------END drawing sample
                        /* for(auto z_it=group_s; z_it<=group_e; z_it++) //updating topic storage
                           training.Zval[z_it]=new_z; */
                        training.Zval[group_s] = new_z;

                        Nw[new_z] += group_count; Nd[new_z] += group_count; Nt[new_z] += group_count;//increament count
                        //------------calcualte the next group start, end index
                        sampled_count += (group_e-group_s+1);
                        //group_count = (size_t) floor(group_count*ratio);// (size_t)floor(ratio*(total_count-sampled_count)); //update group count
                        group_s = group_e+1; 
                        if (group_count==0 || group_count+sampled_count>total_count) {
                            group_count = idx_e-group_s+1; //treat all of what remains as one group
                            //group_count=1;  //treat each of what remains as one group
                        }
                        group_e = group_s+group_count-1;
                    }
                }
            }
        }
        if (iter==1)
        fprintf(log_file, "corpus_total_count=%lu, corpus_sampled_count=%lu , ratio=%.2f\n", corpus_total_count, corpus_sampled_count, ratio);

        size_t singleiter_time = std::clock()-start_time;
        total_cputime += singleiter_time;
        fprintf(log_file, "iter %d LL %.6g cputime %.2f iter-time %.2f\n", iter, compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000), (double)singleiter_time/(CLOCKS_PER_SEC/1000));
        fflush(log_file);
    }

    print_parameters(Nt,Nwt,Ndt, sampler_id);
    //    print_doc_word_topic_triple(training, sampler_id);
    calculate_epsilon(training, test, log_file, Nt, Nwt, Ndt, k, iters, alpha, beta, ratio);
    fclose(log_file);
}

void run_splda_subsampling_iratio(lda_smat_t &training, lda_smat_t &test, const char *src, const char *sampler_id, int k, int iters, double alpha, double beta, double ratio,double iratio) { // {{{
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

    size_t min_set_starttime = std::clock();
    std::unordered_set<size_t> min_set;
    if (exists_min_set(src)) {
        fprintf(log_file, "read from min_set file\n");
        min_set = read_min_set(src);
    }
    else {
        std::set<size_t> tmp = getMinimalDataSet(training);
        min_set.insert(tmp.begin(), tmp.end());
        write_min_set(min_set, src);
        fprintf(log_file, "setcover running time=%.2f\n, set-size=%d, total-size=%d", 
                (double)(std::clock()-min_set_starttime)/(CLOCKS_PER_SEC/1000), min_set.size(), training.nr_docs );
    }

    std::vector<double> Cstatic(k);
    std::vector<entry_t<double> > C(k);
    size_t total_cputime = 0;

    for(int iter = 1; iter <= iters; iter++) {
        size_t nr_A=0, nr_B=0, nr_C=0;
        std::clock_t start_time = clock();
        
        bool iter_uncover = (iter%((int)(iratio*10)))==0;
        size_t corpus_total_count=0, corpus_sampled_count=0;
        for(int iter_counter=0; iter_counter<=1; iter_counter++) {
            if (iter_counter==1 && !iter_uncover) continue;
            for(auto d = 0U; d < training.nr_docs; d++) {
                if(training.doc_ptr[d] == training.doc_ptr[d+1])
                    continue;
                bool inset = min_set.find(d)!=min_set.end();
                if ( iter_counter==0 && !inset)
                    continue;
                if ( iter_counter==1 && inset)
                    continue;

                // per-document variables
                double Asum=0, Bsum=0, Csum=0;
                auto &Nd = Ndt[d];
                for(auto t = 0; t < k; t++) {
                    register double reg_denom = Nt[t]+betabar;
                    register size_t reg_ndt = Nd[t];
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
                        register size_t old_topic = training.Zval[group_s]; 
                        register double reg_denom = Nt[old_topic]+betabar;
                        register size_t reg_ndt = Nd[old_topic];
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

                        register int new_topic=-1;
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

                        Asum -= beta*alpha/reg_denom;
                        if(reg_ndt)
                            Bsum -= beta*reg_ndt/reg_denom;
                        reg_ndt += group_count;
                        reg_denom += group_count;
                        Nt[new_topic] += group_count;
                        Nw[new_topic] += group_count;
                        Nd[new_topic] += group_count;
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
        }
#ifdef DEBUG_COMPRESSION
        if (iter==1)
            fprintf(log_file, "corpus_total_count=%lu, corpus_sampled_count=%lu , ratio=%.2f\n", corpus_total_count, corpus_sampled_count, ratio);
#endif
        size_t singleiter_time = (std::clock()-start_time);
        total_cputime += singleiter_time;
        double all = (double)(nr_A+nr_B+nr_C);
        fprintf(log_file, "iter %d nr_A %.2f nr_B %.2f nz_C %.2f LL %.6g cputime %.2f iter-time %.2f\n", iter, (double)nr_A/all, (double)nr_B/all, (double)nr_C/all, compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000), (double)singleiter_time/(CLOCKS_PER_SEC/1000));
        fflush(log_file);
    }
    print_parameters(Nt,Nwt,Ndt,sampler_id);
    calculate_epsilon(training, test, log_file, Nt, Nwt, Ndt, k, iters, alpha, beta, ratio);
    fclose(log_file);
} // }}}

void run_alias_lda_subsampling_iratio(lda_smat_t &training, lda_smat_t &test, const char *src, const char *sampler_id, int k, int iters, double alpha, double beta, double ratio,double iratio){ //{{{
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

    size_t min_set_starttime = std::clock();
    std::unordered_set<size_t> min_set;
    if (exists_min_set(src)) {
        fprintf(log_file, "read from min_set file\n");
        min_set = read_min_set(src);
    }
    else {
        std::set<size_t> tmp =  getMinimalDataSet(training);
        min_set.insert(tmp.begin(), tmp.end());
        write_min_set(min_set, src);
        fprintf(log_file, "setcover running time=%.2f\n, set-size=%d, total-size=%d", 
                (double)(std::clock()-min_set_starttime)/(CLOCKS_PER_SEC/1000), min_set.size(), training.nr_docs );
    }

    fprintf(log_file, "init LL %.6g\n", compute_training_LL());

    size_t nr_MH_steps = 2;
    size_t total_cputime = 0;
    for(int iter = 1; iter <= iters; iter++) {
        size_t nr_p = 0, nr_q = 0;
        size_t total_MH_steps = 0, rejected_MH_steps = 0;

        bool iter_uncover = (iter%((int)(iratio*10)))==0;
        std::clock_t start_time = clock();
        for(int iter_counter=0; iter_counter<=1; iter_counter++) {
            if (iter_counter==1 && !iter_uncover) continue;

            for(auto d = 0U; d < training.nr_docs; d++) {
                bool inset = min_set.find(d)!=min_set.end();
                if ( iter_counter==0 && !inset)
                    continue;
                if (iter_counter==1 && inset) 
                    continue;

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
                        size_t new_topic = 0;

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
    fclose(log_file);
} // }}}

void run_flda_d_subsampling_iratio(lda_smat_t &training, lda_smat_t &test, const char *src, const char *sampler_id, int k, int iters, double alpha, double beta, double ratio,double iratio){ //{{{
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

    FILE *log_file; 
    char *cp = new char[1000];
    strcpy(cp, sampler_id);
    strcat(cp, ".log.txt");
    log_file = fopen(cp, "w+");

    size_t min_set_starttime = std::clock();
    std::unordered_set<size_t> min_set;
    if (exists_min_set(src)) {
        fprintf(log_file, "read from min_set file\n");
        min_set = read_min_set(src);
    }
    else {
        std::set<size_t> tmp =  getMinimalDataSet(training);
        min_set.insert(tmp.begin(), tmp.end());
        write_min_set(min_set, src);
        fprintf(log_file, "setcover running time=%.2f\n, set-size=%d, total-size=%d", 
                (double)(std::clock()-min_set_starttime)/(CLOCKS_PER_SEC/1000), min_set.size(), training.nr_docs );
    }

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

        bool iter_uncover = (iter%((int)(iratio*10)))==0;

        for(int iter_counter=0; iter_counter<=1; iter_counter++) {
            if (iter_counter==1 && !iter_uncover) continue;
            for(auto d = 0U; d < training.nr_docs; d++) {
                if(training.doc_ptr[d] == training.doc_ptr[d+1]) 
                    continue;

                bool inset = min_set.find(d)!=min_set.end();
                if ( iter_counter==0 && !inset)
                    continue;
                if (iter_counter==1 && inset) 
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
                    size_t group_count = ceil(total_count*ratio); //sample group_count words as a group (assign the same topic to them)
                    // group_count = group_count==0?1:group_count; //total_count
                    size_t sampled_count = 0;
                    auto group_s = idx_s, group_e = idx_s+group_count-1;//group start and end (includsive) index 

                    while (sampled_count < total_count) { //sample one group together
                        // handle each occurrence of word w
                        // Remove counts for the old_topic
                        register size_t old_topic = training.Zval[group_s]; 
                        Nt[old_topic] -= group_count;
                        register double reg_denom = 1.0/(Nt[old_topic]+betabar);
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

                        register int new_topic = -1;
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
                        group_s = group_e+1; 
                        if (group_count+sampled_count>total_count || group_count==0) {
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
    fclose(log_file);

} // }}}

int main(int argc, char *argv[]) {
    //test(); return 0;
    if(argc < 8) {
        puts("[Usage]");
        puts(" $ ./setcover_subsampling nr_topics max_iterations data_dir sampler_id solver ratio iratio");
        puts("solver: 200 = Normal LDA iratio");
        puts("solver: 201 = splda iratio");
        puts("solver: 208 = alias iratio");
        puts("solver: 210 = FLDA-d iratio");
        return -1;
    }
    int k = atoi(argv[1]);
    int iters = atoi(argv[2]);
    char *src = argv[3];
    int solver = atoi(argv[4]);
    char *sampler_id = argv[5];
    double alpha = 50.0/k, beta = 0.01;
    double ratio = atof(argv[6]); 
    double iratio = atof(argv[7]); 
    double uratio;
    if(argc==9)
        uratio=atof(argv[8]); 

    // srand(time(NULL));
    // srand48(time(NULL));

    lda_smat_t training_set, test_set;
    lda_read_data(src, training_set, test_set, 1, smat_t::PETSc);
    //printf("%s\n",src);
    switch (solver) {
        case 200 :
            run_lda_subsampling_iratio(training_set, test_set, src, sampler_id, k, iters, alpha, beta, ratio, iratio);
            break;
        case 201 :
            run_splda_subsampling_iratio(training_set, test_set, src, sampler_id, k, iters, alpha, beta, ratio, iratio);
            break;
        case 208 :
            run_alias_lda_subsampling_iratio(training_set, test_set, src, sampler_id, k, iters, alpha, beta, ratio, iratio);
            break;
        case 210 :
            run_flda_d_subsampling_iratio(training_set, test_set, src, sampler_id, k, iters, alpha, beta, ratio, iratio);
            break;
    }
    return 0;
}
