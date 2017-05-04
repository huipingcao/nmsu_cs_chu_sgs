//#define DEBUG_GROUPS
#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <map>
#include <utility>
#include <assert.h>
#include <iostream>
#include <fstream>

#include "splda.h"
#include "sub_lda.h"
#include "sub_splda.h"
#include "sub_alias_lda.h"
#include "sub_flda_d.h"

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
}//}}}

// Sparse-sampling with 3 terms (used in Yahoo LDA)
void run_splda(lda_smat_t &training, lda_smat_t &test, char* sampler_id, int k, int iters, double alpha, double beta) { // {{{
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
                for(auto Zidx = training.Z_ptr[idx]; Zidx != training.Z_ptr[idx+1]; Zidx++) {
                    // handle each occurrence of word w
                    register size_t old_topic = training.Zval[Zidx]; 
                    register double reg_denom = Nt[old_topic]+betabar;
                    register size_t reg_ndt = Nd[old_topic]--;

                    // removal old_topic
                    Asum -= beta*alpha/reg_denom;
                    if(reg_ndt)
                        Bsum -= beta*reg_ndt/reg_denom;
                    --reg_ndt;
                    --reg_denom;
                    --Nt[old_topic];
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
                            Nw[t]--;
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
                    // sampling new topic // {{{
                    if(sample < Csum) {
                        auto *ptr = C.data();
                        while((sample-=ptr->value) > 0)
                            ptr++;
                        new_topic = ptr->idx;
                        nr_C++;
                    } else {
                        sample -= Csum;
                        if(sample < Bsum) {
                            sample /= beta;
                            for(auto &t : Nd.nz_idx) {
                                sample -= Nd[t]/(Nt[t]+betabar);
                                if(sample < 0) {
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
                                if(sample < 0) {
                                    new_topic = t;
                                    break;
                                }
                            }
                            nr_A++;
                        }
                    } // }}}
                training.Zval[Zidx] = new_topic;

                // Add counts for the new_topic
                reg_denom = Nt[new_topic]+betabar;
                reg_ndt = Nd[new_topic]++;

                Asum -= beta*alpha/reg_denom;
                if(reg_ndt)
                    Bsum -= beta*reg_ndt/reg_denom;
                ++reg_ndt;
                ++reg_denom;
                ++Nt[new_topic];
                ++Nw[new_topic];
                if(reg_ndt==1)
                    Nd.push(new_topic);
                if(Nw[new_topic]==1)
                    Nw.push(new_topic);
                Asum += beta*alpha/reg_denom;
                Bsum += beta*reg_ndt/reg_denom;
                Cstatic[new_topic] = (reg_ndt+alpha)/reg_denom;
                }
            }
        }
        size_t singleiter_time = (std::clock()-start_time);
        total_cputime += singleiter_time;
        double all = (double)(nr_A+nr_B+nr_C);
        fprintf(log_file, "iter %d nr_A %.2f nr_B %.2f nz_C %.2f LL %.6g cputime %.2f iter-time %.2f\n", iter, (double)nr_A/all, (double)nr_B/all, (double)nr_C/all, compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000), (double)singleiter_time/(CLOCKS_PER_SEC/1000) );
        fflush(log_file);
    }
    fprintf(log_file, "push_call_count=%d, remove_call_count=%d\n", push_call_count, remove_call_count);
    fclose(log_file);
    print_parameters(Nt,Nwt,Ndt,sampler_id);

} // }}}
// F+LDA Sampling with 2 terms doc-by-doc (best for now) _test
void run_splda_fast_2_test(lda_smat_t &training, lda_smat_t &test, char *sampler_id, int k, int iters, double alpha, double beta) { // {{{
    
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
                for(auto Zidx = training.Z_ptr[idx]; Zidx != training.Z_ptr[idx+1]; Zidx++) {
                    // handle each occurrence of word w
                    // Remove counts for the old_topic
                    register size_t old_topic = training.Zval[Zidx]; 
                    register double reg_denom = 1.0/((--Nt[old_topic])+betabar);
                    --Nw[old_topic];--Nd[old_topic];
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
                    training.Zval[Zidx] = new_topic;
                    assert(new_topic != -1 && new_topic < k);

                    // Add counts for the new_topic
                    reg_denom = 1./((++Nt[new_topic])+betabar);
                    ++Nd[new_topic]; ++Nw[new_topic];

                    if((int)old_topic != new_topic) {
                        if(Nd[old_topic]==0) Nd.pop(old_topic);
                        if(ptr_Nw_old_topic) Nw.remove_ptr(ptr_Nw_old_topic);
                        if(Nw[new_topic]==1) Nw.push(new_topic);
                        if(Nd[new_topic]==1) Nd.push(new_topic);
                        if(not updated) D.update_parent(old_topic, D_new-D_old);
                        D.set_value(new_topic, reg_denom*(alpha+Nd[new_topic]));
                    } else {
                        //if(not updated) D.update_parent(old_topic, D_new-D_old);
                        //D.set_value(new_topic, reg_denom*(beta+Nw[new_topic]));
                        if(updated) D.set_value(old_topic, D_old);
                        else D.true_val[old_topic] = D_old;
                    }

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
        size_t singleiter_time = (std::clock()-start_time);
        total_cputime += singleiter_time;
        double all = (double)(nr_A+nr_B+nr_C+nr_D);
        fprintf(log_file,"iter %d nr_C %.2f nr_D %.2f LL %.6g cputime %.2f iter-time %.2f\n",
                iter, (double)nr_C/all, (double)nr_D/all, 
                compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000), (double)singleiter_time/(CLOCKS_PER_SEC/1000));
        fflush(log_file);
    }

    fprintf(log_file, "push_call_count=%d, remove_call_count=%d\n", push_call_count, remove_call_count);
    print_parameters(Nt,Nwt,Ndt,sampler_id);
    fclose(log_file);
} // }}}

// F+LDA Sampling with 2 terms word-by-word(best for now) _test
void run_splda_fast_2_word_test(lda_smat_t &training, lda_smat_t &test, char *sampler_id, int k, int iters, double alpha, double beta) { // {{{
    
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
    training.initialize_Z_wordwise(k);
    std::vector<size_t> nnz_w(training.nr_words), nnz_d(training.nr_docs);
    size_t	nnz_t = 0;

    for(auto w = 0U; w < training.nr_words; w++) {
        for(auto idx = training.word_ptr[w]; idx != training.word_ptr[w+1]; idx++) {
            auto d = training.doc_idx[idx];
            for(auto zidx = training.Z_ptr[idx]; zidx != training.Z_ptr[idx+1]; zidx++) {
                auto t = training.Zval[zidx];
                Nt[t]++;
                Ndt[d][t]++;
                Nwt[w][t]++;
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
        D[t] = beta/(Nt[t]+betabar);
    D.init_dense();
    total_cputime += clock() - tmp_start;
    for(int iter = 1; iter <= iters; iter++) {
        size_t nr_A=0, nr_B=0, nr_C=0, nr_D=0;
        std::clock_t start_time = clock();
        for(auto w = 0U; w < training.nr_words; w++) {
            if(training.word_ptr[w] == training.word_ptr[w+1])
                continue;
            // per-word variables (D)
            auto &Nw = Nwt[w];
            if(Nw.nz_idx.size() > threshold){
                for(auto t: Nw.nz_idx) {
                    //double denom = 1.0/(Nt[t]+betabar); D.true_val[t] = (beta+Nw[t])*denom;
                    D.true_val[t] = (beta+Nw[t])/(Nt[t]+betabar);
                }
                D.init_dense();
            } else  {
                for(auto t: Nw.nz_idx) {
                    //double denom = 1.0/(Nt[t]+betabar); D.set_value(t, (beta+Nw[t])*denom);
                    D.set_value(t, (beta+Nw[t])/(Nt[t]+betabar));
                }
            }
            /*
               D.true_val[t] = (alpha+Nd[t])/(Nt[t]+betabar);
               D.init_sparse(Nd.nz_idx.data(), Nd.nz_idx.size());
               */

            for(auto idx = training.word_ptr[w]; idx != training.word_ptr[w+1]; idx++) {
                auto d = training.doc_idx[idx];
                auto &Nd = Ndt[d];
                for(auto Zidx = training.Z_ptr[idx]; Zidx != training.Z_ptr[idx+1]; Zidx++) {
                    // handle each occurrence of doc d
                    // Remove counts for the old_topic
                    register size_t old_topic = training.Zval[Zidx]; 
                    register double reg_denom = 1.0/((--Nt[old_topic])+betabar);
                    --Nw[old_topic];--Nd[old_topic];
                    double D_old = D.true_val[old_topic];
                    double D_new = reg_denom*(beta+Nw[old_topic]);
                    D.true_val[old_topic] = D_new; 
                    bool updated = false;
                    //D.set_value(old_topic, reg_denom*(beta+Nw[old_topic]));
                    //D.update_parent(old_topic, D_new-D_old);
                    //if(Nw[old_topic]==0) Nw.pop(old_topic);

                    // Handle Inner Product (Part C) {{{
                    size_t nz_C = 0;
                    size_t *ptr_Nd_old_topic = NULL;
                    double Csum = 0;
                    for(auto &t : Nd.nz_idx) {
                        if(Nd[t]) {
                            /*
                               C[nz_C].idx = t;
                            //C[nz_C].value = Nw[t]*(Nd[t]+alpha)*D[t];
                            //C[nz_C].value = Nw[t]*(B[t]+alpha*D[t]);
                            //C[nz_C].value = Nw[t]*(B[t]+D[t]);
                            //C[nz_C].value = Nw[t]*(B[t]);
                            //C[nz_C].value = A[t]*Nd[t];
                            C[nz_C].value = Nd[t]*D.true_val[t];
                            Csum += C[nz_C].value;
                            */
                            C[nz_C] = (Csum += Nd[t]*D.true_val[t]);
                            C_idx[nz_C] = t;
                            nz_C++;
                        } else {
                            ptr_Nd_old_topic = &t;
                        }
                    }
                    //if(ptr_Nd_old_topic) Nd.remove_ptr(ptr_Nd_old_topic);
                    // }}}

                    register int new_topic = -1;
                    double Dsum = Csum+alpha*(D.total_sum()-D_old+D_new);
                    //double Dsum = Csum+alpha*D.total_sum();
                    double sample = drand48()*Dsum; 
                    //printf("sample %g Csum %g Asum %g Bsum %g Dsum %g\n", sample, Csum, Asum, Bsum, Dsum);
                    if(sample < Csum) { // {{{
                        auto *ptr = C.data();
                        new_topic = C_idx[std::upper_bound(ptr, ptr+nz_C, sample)-ptr];
                        /*
                           while((sample-=ptr->value) > 0)
                           ptr++;
                           new_topic = ptr->idx;
                           */
                        nr_C++;
                    } else {
                        sample = (sample-Csum)/alpha;
                        D.update_parent(old_topic, D_new-D_old);
                        updated = true;
                        new_topic = D.log_sample(sample);
                        nr_D++;
                    } // }}}
                    training.Zval[Zidx] = new_topic;
                    //assert(new_topic != -1 && new_topic < k);

                    // Add counts for the new_topic
                    reg_denom = 1./((++Nt[new_topic])+betabar);
                    ++Nd[new_topic]; ++Nw[new_topic];
                    //if(not updated) D.update_parent(old_topic, D_new-D_old);
                    //D.set_value(new_topic, reg_denom*(beta+Nw[new_topic]));
                    if((int)old_topic != new_topic) {
                        if(Nw[old_topic]==0) Nw.pop(old_topic);
                        if(ptr_Nd_old_topic) Nd.remove_ptr(ptr_Nd_old_topic);
                        if(Nw[new_topic]==1) Nw.push(new_topic);
                        if(Nd[new_topic]==1) Nd.push(new_topic);
                        if(not updated) D.update_parent(old_topic, D_new-D_old);
                        D.set_value(new_topic, reg_denom*(beta+Nw[new_topic]));
                    } else {
                        //if(not updated) D.update_parent(old_topic, D_new-D_old);
                        //D.set_value(new_topic, reg_denom*(beta+Nw[new_topic]));
                        if(updated) D.set_value(old_topic, D_old);
                        else D.true_val[old_topic] = D_old;
                    }
                }
            }
            if(Nw.nz_idx.size() > threshold){
                for(auto t: Nw.nz_idx)
                    D.true_val[t] = beta/(Nt[t]+betabar);
                D.init_dense();
            } else  {
                for(auto t: Nw.nz_idx)
                    D.set_value(t, beta/(Nt[t]+betabar));
            }

            /*
               D.true_val[t] = alpha/(Nt[t]+betabar);
               D.init_sparse(Nd.nz_idx.data(), Nd.nz_idx.size());
               */
        }
        size_t singleiter_time = (std::clock()-start_time);
        total_cputime += singleiter_time;
        double all = (double)(nr_A+nr_B+nr_C+nr_D);
        // printf("iter %d nr_C %.2f nr_D %.2f LL %.6g cputime %.2f iter-time %.2f init(%.2f) update(%.2f) sample(%.2f)\n",
        //         iter, (double)nr_C/all, (double)nr_D/all, 
        //         compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000),
        //         (double)singleiter_time/(CLOCKS_PER_SEC/1000),
        //         D.get_init_time(),  D.get_update_time(), D.get_sample_time());
        fprintf(log_file, "iter %d nr_C %.2f nr_D %.2f LL %.6g cputime %.2f iter-time %.2f init(%.2f) update(%.2f) sample(%.2f)\n",
                iter, (double)nr_C/all, (double)nr_D/all, 
                compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000),
                (double)singleiter_time/(CLOCKS_PER_SEC/1000));

        fflush(log_file);
    }
    print_parameters(Nt,Nwt,Ndt,sampler_id);
    fclose(log_file);

} // }}}

void run_lda_llh_iter_check(lda_smat_t &training, lda_smat_t &test, char *sampler_id, int k, int iters, double alpha, double beta){ //{{{
    typedef std::vector<unsigned> vec_t;
    typedef std::vector<vec_t> mat_t;

    vec_t Nt(k, 0);
    mat_t Nwt(training.nr_words, vec_t(k));
    mat_t Ndt(training.nr_docs, vec_t(k));
    training.initialize_Z_docwise(k);

    for(auto d = 0U; d <training.nr_docs; d++) {
        for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
            auto w = training.word_idx[idx];
            //printf("%lu \n",w);
            for(auto zidx = training.Z_ptr[idx]; zidx != training.Z_ptr[idx+1]; zidx++) {
                auto t = training.Zval[zidx];
                Nt[t]++;
                Ndt[d][t]++;
                Nwt[w][t]++;
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
    auto sampler = [&](double sum)->int {
        double urnd = drand48()*sum; int t = 0;
        while(1) {
            urnd -= params[t];
            if(urnd < 0) break;
            t++;
        }
        return t;
    };
    
    FILE *log_file; 
    char *cp = new char[1000];
    strcpy(cp, sampler_id);
    strcat(cp, ".log.txt");
    log_file = fopen(cp, "w+");

    fprintf(log_file, "init LL %.6g\n", compute_training_LL());
    size_t total_cputime = 0;
    for(int iter = 1; iter <= iters; iter++) {
        std::clock_t start_time = clock();
        for(auto d = 0U; d < training.nr_docs; d++) {
            auto &Nd = Ndt[d];
            for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
                auto w = training.word_idx[idx];
                auto &Nw = Nwt[w];
                for(auto Zidx = training.Z_ptr[idx]; Zidx != training.Z_ptr[idx+1]; Zidx++) {
                    // #ifdef DEBUG_SUBSAMPLE
                    //printf("d=%lu w=%lu\n", d, w);
                    // #endif
                    size_t old_topic = training.Zval[Zidx];
                    Nw[old_topic]--;
                    Nd[old_topic]--;
                    Nt[old_topic]--;
                    double sum = 0;
                    for(auto t = 0; t < k; t++) {
                        params[t] = (alpha+Nd[t])*(beta+Nw[t])/(betabar+Nt[t]);
                        sum += params[t];
                    }
                    size_t new_topic = sampler(sum);
                    training.Zval[Zidx] = new_topic;
                    Nw[new_topic]++;
                    Nd[new_topic]++;
                    Nt[new_topic]++;
                }
            }
            fprintf(log_file, "doc %d LL %.6g\n", d, compute_training_LL());
        }
        size_t singleiter_time = std::clock()-start_time;
        total_cputime += singleiter_time;
        fprintf(log_file, "iter %d LL %.6g cputime %.2f iter-time %.2f\n", iter, compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000), (double)singleiter_time/(CLOCKS_PER_SEC/1000));
        fflush(log_file);
    }
    fclose(log_file);
}

// Normal sampling doc-by-doc 
void run_lda(lda_smat_t &training, lda_smat_t &test, char *sampler_id, int k, int iters, double alpha, double beta){ //{{{
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
    auto sampler = [&](double sum)->int {
        double urnd = drand48()*sum; int t = 0;
        while(1) {
            urnd -= params[t];
            if(urnd < 0) break;
            t++;
        }
        return t;
    };

    FILE *log_file; 
    char *cp = new char[1000];
    strcpy(cp, sampler_id);
    strcat(cp, ".log.txt");
    log_file = fopen(cp, "w+");

    fprintf(log_file, "init LL %.6g\n", compute_training_LL());
    size_t total_cputime = 0;
    for(int iter = 1; iter <= iters; iter++) {
        std::clock_t start_time = clock();
        for(auto d = 0U; d < training.nr_docs; d++) {
            auto &Nd = Ndt[d];
            for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
                auto w = training.word_idx[idx];
                auto &Nw = Nwt[w];
                for(auto Zidx = training.Z_ptr[idx]; Zidx != training.Z_ptr[idx+1]; Zidx++) {
                    size_t old_topic = training.Zval[Zidx];
                    Nw[old_topic]--;
                    Nd[old_topic]--;
                    Nt[old_topic]--;
                    double sum = 0;
                    for(auto t = 0; t < k; t++) {
                        params[t] = (alpha+Nd[t])*(beta+Nw[t])/(betabar+Nt[t]);
                        sum += params[t];
                    }
                    size_t new_topic = sampler(sum);
                    training.Zval[Zidx] = new_topic;
                    Nw[new_topic]++;
                    Nd[new_topic]++;
                    Nt[new_topic]++;
                }
            }
        }
        size_t singleiter_time = std::clock()-start_time;
        total_cputime += singleiter_time;
        fprintf(log_file, "iter %d LL %.6g cputime %.2f iter-time %.2f\n", iter, compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000), (double)singleiter_time/(CLOCKS_PER_SEC/1000));
        fflush(log_file);
    }

    print_doc_word_topic_triple(training, sampler_id); //TODO comment later
    print_parameters(Nt,Nwt,Ndt, sampler_id);
    fclose(log_file);
} // }}}

// Normal sampling word-by-word
void run_lda_word(lda_smat_t &training, lda_smat_t &test, int k, int iters, double alpha, double beta){ //{{{
    typedef std::vector<size_t> vec_t;
    typedef std::vector<vec_t> mat_t;
    vec_t Nt(k, 0);
    mat_t Nwt(training.nr_words, vec_t(k,0));
    mat_t Ndt(training.nr_docs, vec_t(k,0));
    training.initialize_Z_wordwise(k);

    for(auto w = 0U; w < training.nr_words; w++) {
        for(auto idx = training.word_ptr[w]; idx != training.word_ptr[w+1]; idx++) {
            auto d = training.doc_idx[idx];
            for(auto zidx = training.Z_ptr[idx]; zidx != training.Z_ptr[idx+1]; zidx++) {
                auto t = training.Zval[zidx];
                Nt[t]++;
                Ndt[d][t]++;
                Nwt[w][t]++;
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
    auto sampler = [&](double sum)->int {
        double urnd = drand48()*sum; int t = 0;
        while(1) {
            urnd -= params[t];
            if(urnd < 0) break;
            t++;
        }
        return t;
    };

    printf("init LL %.6g\n", compute_training_LL());
    size_t total_cputime = 0;
    for(int iter = 1; iter <= iters; iter++) {
        std::clock_t start_time = clock();
        for(auto w = 0U; w < training.nr_words; w++) {
            auto &Nw = Nwt[w];
            for(auto idx = training.word_ptr[w]; idx != training.word_ptr[w+1]; idx++) {
                auto d = training.doc_idx[idx];
                auto &Nd = Ndt[d];
                for(auto Zidx = training.Z_ptr[idx]; Zidx != training.Z_ptr[idx+1]; Zidx++) {
                    size_t old_topic = training.Zval[Zidx];
                    Nw[old_topic]--;
                    Nd[old_topic]--;
                    Nt[old_topic]--;
                    double sum = 0;
                    for(auto t = 0; t < k; t++) {
                        params[t] = (alpha+Nd[t])*(beta+Nw[t])/(betabar+Nt[t]);
                        sum += params[t];
                    }
                    size_t new_topic = sampler(sum);
                    training.Zval[Zidx] = new_topic;
                    Nw[new_topic]++;
                    Nd[new_topic]++;
                    Nt[new_topic]++;
                }
            }
        }
        total_cputime += std::clock()-start_time;
        printf("iter %d LL %.6g cputime %.2f\n", iter, compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000));
        fflush(stdout);
    }
} // }}}

// Alias LDA
void run_alias_lda(lda_smat_t &training, lda_smat_t &test, char* sampler_id, int k, int iters, double alpha, double beta){ //{{{
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

    for(auto &Nd: Ndt) Nd.gen_nz_idx();

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
                for(auto Zidx = training.Z_ptr[idx]; Zidx != training.Z_ptr[idx+1]; Zidx++) {
                    size_t old_topic = training.Zval[Zidx];
                    Nw[old_topic]--;
                    Nd[old_topic]--;
                    Nt[old_topic]--;
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
                    training.Zval[Zidx] = new_topic;
                    Nw[new_topic]++;
                    Nd[new_topic]++;
                    Nt[new_topic]++;
                    if(Nd[new_topic]==1) Nd.push(new_topic);
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
    fprintf(log_file, "push_call_count=%d, remove_call_count=%d\n", push_call_count, remove_call_count);
    print_parameters(Nt,Nwt,Ndt,sampler_id);
    fclose(log_file);
} // }}}

int main(int argc, char *argv[]) {
    //test(); return 0;
    if(argc < 6) {
        puts("[Usage]");
        puts(" $ ./splda nr_topics max_iterations data_dir solver sampler_id");
        puts("solver: 0 = Normal LDA");
        puts("solver: 1 = Sparse LDA");
        puts("solver: 8 = Alias LDA");
        puts("solver: 9 = F+LDA - word-by-word");
        puts("solver: 10 = F+LDA - doc-by-doc");
        puts("solver: 100 = subsampling Normal LDA");
        return -1;
    }
    int k = atoi(argv[1]);
    int iters = atoi(argv[2]);
    char *src = argv[3];
    int solver = atoi(argv[4]);
    char *sampler_id = argv[5];

    double alpha = 50.0/k, beta = 0.01;
    double ratio = 0.5;
    if(argc==7)
        ratio = std::stod(argv[6]);

    //double alpha = 0.1, beta = 0.1;
    // srand(time(NULL));
    // srand48(time(NULL));

    lda_smat_t training_set, test_set;
    lda_read_data(src, training_set, test_set, 1, smat_t::PETSc);
    //printf("%s\n",src);
    switch (solver) {
        case -1 :
            run_lda_llh_iter_check(training_set, test_set, sampler_id, k, iters, alpha, beta);
            break;
        case 0 :
            run_lda(training_set, test_set, sampler_id, k, iters, alpha, beta);
            break;
        case 1 :
            run_splda(training_set, test_set, sampler_id, k, iters, alpha, beta);
            break;
        case 2 :
            training_set.initialize_Z_docwise(k);
            get_word_groups(training_set,k);
            break;
        case 8 :
            run_alias_lda(training_set, test_set, sampler_id, k, iters, alpha, beta);
            break;
        case 9 :
            run_splda_fast_2_word_test(training_set, test_set, sampler_id, k, iters, alpha, beta);
            break;
        case 10 :
            run_splda_fast_2_test(training_set, test_set, sampler_id, k, iters, alpha, beta);
            break;
        case 100:
            run_lda_subsampling(training_set, test_set, sampler_id, k, iters, alpha, beta, ratio);
            break;
        case 101:
            run_splda_subsampling(training_set, test_set, sampler_id, k, iters, alpha, beta, ratio);
            break;
        case 108:
            run_alias_lda_subsampling(training_set, test_set, sampler_id, k, iters, alpha, beta, ratio);
            break;
        case 110:
            run_splda_fast_2_test_subsampling(training_set, test_set, sampler_id, k, iters, alpha, beta, ratio);
            break;
    }

    printf("solver=%d, push_call_count=%d, remove_call_count=%d\n", solver, push_call_count, remove_call_count);

    return 0;
}
