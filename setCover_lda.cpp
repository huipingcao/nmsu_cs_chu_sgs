#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <map>
#include <utility>
#include <assert.h>
#include <iostream>

#include "splda.h"
#include "SetCover.h"
#define PRINT_DOC

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

void run_lda(lda_smat_t &training, lda_smat_t &test, int k, int iters, double alpha, double beta){ //{{{
    //printf("run_lda function \n");
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
    

    size_t total_cputime = 0;
    printf("***************************\n");
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
#ifdef PRINT_DOC
            printf("doc %d LL %.6g\n", d, compute_training_LL());
#endif
        }

        size_t singleiter_time = std::clock()-start_time;
        total_cputime += singleiter_time;
        printf("iter %d LL %.6g cputime %.2f iter-time %.2f\n", iter, compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000), (double)singleiter_time/(CLOCKS_PER_SEC/1000));
        fflush(stdout);
    }
}
void run_lda_setcover_llh_iter_check(lda_smat_t &training, lda_smat_t &test, int k, int iters, double alpha, double beta, double ratio){ //{{{
    //printf("run_lda function \n");
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

    size_t min_set_starttime = std::clock();
    std::set<size_t> minSet;
    minSet = getMinimalDataSet(training);
    printf("setcover running time=%.2f\n, set-size=%d, total-size=%d", 
            (double)(std::clock()-min_set_starttime)/(CLOCKS_PER_SEC/1000), minSet.size(), training.nr_docs );

    size_t total_cputime = 0;
    printf("***************************\n");
    for(int iter = 1; iter <= iters; iter++) {
        std::clock_t start_time = clock();
        for(auto d = 0U; d < training.nr_docs; d++) {
            if(minSet.find(d)!=minSet.end()){
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
#ifdef PRINT_DOC
                printf("doc %d LL %.6g\n", d, compute_training_LL());
#endif
            }
        }
        if ( ( iter%((int)(ratio*10)) )==0 ) {
            for(auto d = 0U; d < training.nr_docs; d++) {
                if(minSet.find(d)==minSet.end()){
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
#ifdef PRINT_DOC
                    printf("doc %d LL %.6g\n", d, compute_training_LL());
#endif
                }
            }
        }

        size_t singleiter_time = std::clock()-start_time;
        total_cputime += singleiter_time;
        printf("iter %d LL %.6g cputime %.2f iter-time %.2f\n", iter, compute_training_LL(), (double)total_cputime/(CLOCKS_PER_SEC/1000), (double)singleiter_time/(CLOCKS_PER_SEC/1000));
        fflush(stdout);
    }
} // }}}


int main(int argc, char *argv[]) {
    int k = atoi(argv[1]);
    int iters = atoi(argv[2]);
    char *src = argv[3];
    int solver = atoi(argv[4]);
    double alpha = 50.0/k, beta = 0.01;
    double ratio = atof(argv[5]); 
    double iratio = atof(argv[6]); 

    //double alpha = 0.1, beta = 0.1;
    // srand(time(NULL));
    // srand48(time(NULL));

    lda_smat_t training_set, test_set;
    lda_read_data(src, training_set, test_set, 1, smat_t::PETSc);
    //printf("%s\n",src);
    switch (solver) {
        case 0 :
            run_lda(training_set, test_set, k, iters, alpha, beta);
            break;
        case -1 :
            run_lda_setcover_llh_iter_check(training_set, test_set, k, iters, alpha, beta, ratio);
            break;
    }

    return 0;
}
