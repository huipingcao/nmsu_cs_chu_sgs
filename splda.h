#ifndef SPLDA_H
#define SPLDA_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <vector>
#include <set>
#include <array>
#include <tuple>
#include <random>
#ifndef _GLIBCXX_USE_SCHED_YIELD
#define _GLIBCXX_USE_SCHED_YIELD
#endif
#ifndef _GLIBCXX_USE_NANOSLEEP
#define _GLIBCXX_USE_NANOSLEEP
#endif
#include <thread>
#include <cmath>
#include <chrono>
#include <functional> // for std::ref used for std::thread
#include <unistd.h> // access for existence
#include <omp.h>

extern unsigned int push_call_count = 0; //TODO profile memory intensive function call only. remove later
extern unsigned int remove_call_count = 0; //TODO profile memory intensive function call only. remove later
/*
#include "tbb/atomic.h"
#include "tbb/concurrent_queue.h"
#include "tbb/concurrent_vector.h"
#include "tbb/scalable_allocator.h"
#include "tbb/cache_aligned_allocator.h"
*/
#include <atomic>
#include "sparse_matrix.h"
#include "petsc-reader.h"

//#include "serialize.h"

typedef std::mt19937 rng_type;

template <typename T> 
//using sallocator = tbb::scalable_allocator<T>;
using sallocator = std::allocator<T>;

template <typename T> 
//using callocator = tbb::cache_aligned_allocator<T>;
using callocator = std::allocator<T>;

//template <typename T>
//using con_queue = tbb::concurrent_queue<T>;

template <typename T>
//using atomic = tbb::atomic<T>;
using atomic = std::atomic<T>;

class dist_lda_param_t {//{{{
	public:
		enum{load_balance =0, rand_perm = 1};

		long k;
		double alpha, beta;
		int delay;
		int samplers;
		int nr_procs;
		double time_interval;
		int schedule;
		int maxinneriter;
		int maxiter;
		int max_tokens_per_msg;
		int perplexity_samples;
		int lrate_method, num_blocks; 
		int do_predict, verbose;
		dist_lda_param_t() {
			k = 10;
			alpha = 5;
			beta = 0.01;
			delay = 10;
			samplers = 4;
			nr_procs = 1;
			maxinneriter = 5;
			time_interval = 5.0;
			schedule = load_balance;
			maxiter = 5;
			max_tokens_per_msg = 20;
			perplexity_samples = 100;
			do_predict = 0;
			verbose = 0;
		}
};//}}}

typedef int wc_t; // type for word_count
typedef sparse_matrix<wc_t> smat_t; // sparse word count matrix
typedef smat_subset_iterator_t<wc_t> smat_subset_it; // subset iterator
typedef PETSc_reader<wc_t> petsc_reader_t; // PETSC_reader

// for topic-model
typedef long var_t; // variable type, size_t for topic_count, the main type used in the model

#if defined( RERM_64BIT )
	typedef int64_t IndexType;
#define MPIU_INT MPI_LONG_LONG_INT
#else
	typedef int IndexType;
#define MPIU_INT MPI_INT
#endif

#define MPI_VAR_T MPI_INT64_T

// divide docs into nr_procs*nr_samplers blocks.
class doc_partitioner_t: public std::vector<int> { //{{{
	public:
		size_t nr_docs, nr_blocks, blocksize;
		int nr_procs, nr_samplers;
		doc_partitioner_t(size_t nr_docs_ = 0, int nr_procs_ = 0, int nr_samplers_ = 0) {
			if(nr_procs_ != 0 and nr_samplers_ != 0)
				init(nr_docs_, nr_procs_, nr_samplers_);
		}
		void init(size_t nr_docs_, int nr_procs_, int nr_samplers_) {
			nr_docs = nr_docs_; nr_procs = nr_procs_; nr_samplers = nr_samplers_;
			nr_blocks = nr_procs * nr_samplers;
			blocksize = nr_docs/nr_blocks + ((nr_docs%nr_blocks)?1:0);
			this->resize(nr_blocks+1);
			for(auto i = 1U; i < nr_blocks; i++)
				this->at(i) = this->at(i-1) + blocksize;
			this->at(nr_blocks) = nr_docs;
		}
		size_t start_doc(int procid, int samplerid) {
			return this->at(procid*nr_samplers + samplerid);
		}
		size_t end_doc(int procid, int samplerid) {
			return this->at(procid*nr_samplers+samplerid+1);
		}
};//}}}

class lda_smat_t { // {{{
	public:
		smat_t word_count;
		size_t nr_docs, nr_words, start_doc, end_doc, nnz, Z_len;
		size_t *word_ptr, *doc_ptr, *Z_ptr;
		unsigned *doc_idx, *word_idx, *Zval;
		wc_t *val, *val_t;

		//size_t* &col_ptr; //=word_ptr;
		//unsigned* &row_idx; //=doc_idx;

		bool mem_alloc_by_me;

		lda_smat_t(){
			nr_docs = nr_words = start_doc = end_doc = Z_len = 0;
			doc_ptr = word_ptr = Z_ptr = NULL;
			word_idx = doc_idx = Zval = NULL;
			mem_alloc_by_me = false;
			//col_ptr = word_ptr; row_idx = doc_idx;
		}
		lda_smat_t(const lda_smat_t &Z){*this=Z; mem_alloc_by_me=false;}

		~lda_smat_t() { clear_space(); }

		void load_data(size_t nr_docs_, size_t nr_words_, size_t nnz_, 
				const char *filename, smat_t::format_t file_fmt) {
					
		    //printf("parameter in lad_smat_t load_data() function is %lu %lu %lu %s %lu \n", nr_docs_, nr_words_, nnz_, filename, file_fmt);
			word_count.load(nr_docs_, nr_words_, nnz_, filename, file_fmt);
			fill_attributes();
			//printf("----------------------------------------\n %lu \n",val[746315]);

		}

		void load_data(smat_subset_it it) {
			word_count.load_from_iterator(it.get_rows(), it.get_cols(), it.get_nnz(), &it);
			fill_attributes();
		}

		void load_data(petsc_reader_t &reader, size_t start_row, size_t end_row, std::vector<int> *col_perm=NULL) {
			reader.receive_rows(start_row, end_row, word_count, col_perm);
			fill_attributes();
			start_doc = start_row; end_doc = end_row;
		}

		void fill_attributes() {
			nr_docs = word_count.rows; nr_words = word_count.cols; nnz = word_count.nnz;
			start_doc = 0; end_doc = nr_docs;
			word_ptr = word_count.col_ptr;
			doc_idx = word_count.row_idx;
			val = word_count.val;

			doc_ptr = word_count.row_ptr;
			word_idx = word_count.col_idx;
			val_t = word_count.val_t;
		}

		void initialize_Z_wordwise(int k, unsigned seed = 0) {
			Z_ptr = sallocator<size_t>().allocate(word_count.nnz+1);
			if(Z_ptr == NULL) {
				puts("No enough memory for Zptr\n");
			}
			Z_len = 0; Z_ptr[0] = 0;
			for(size_t idx = 0; idx < word_count.nnz; idx++) {
				Z_len += word_count.val[idx]; // we want CSC ordering
				Z_ptr[idx+1] = Z_len; 
			}
			printf("z_len:%lu \n",Z_len);
			Zval = sallocator<unsigned>().allocate(Z_len);
			if(Zval == NULL) {
				puts("No enough memory for Zval\n");
			}
			for(size_t zidx = 0; zidx < Z_len; zidx++)
				Zval[zidx] = rand_r(&seed) % k;
			mem_alloc_by_me = true;
		}

		void initialize_Z_docwise(int k, unsigned seed = 0) {
			//printf("-------------------------------------\n");
			//printf("initialize_Z_docwise \n");
			Z_ptr = sallocator<size_t>().allocate(word_count.nnz+1);
			if(Z_ptr == NULL) {
				puts("No enough memory for Zptr");
			}
			Z_len = 0; Z_ptr[0] = 0;
			for(size_t idx = 0; idx < word_count.nnz; idx++) {
				Z_len += word_count.val_t[idx]; // we want CSR ordering
				Z_ptr[idx+1] = Z_len; 
			}
			
            printf("INIT: total word count=%lu\n", Z_len);
			//printf("z_len:%lu \n",Z_len);

			
			Zval = sallocator<unsigned>().allocate(Z_len);
			if(Zval == NULL) {
				puts("No enough memory for Zval\n");
			}
			for(size_t zidx = 0; zidx < Z_len; zidx++)
				Zval[zidx] = rand_r(&seed) % k;
			mem_alloc_by_me = true;
		}

		void clear_space(){
			if(mem_alloc_by_me) {
				if(Z_ptr) sallocator<size_t>().deallocate(Z_ptr, word_count.nnz+1);
				if(Zval) sallocator<unsigned>().deallocate(Zval, Z_len);
				//if(Z_ptr) free(Z_ptr); //sallocator<size_t>().deallocate(Z_ptr, word_count.nnz+1);
				//if(Zval) free(Zval); //sallocator<unsigned>().deallocate(Zval, Z_len);
			}
		}

};//}}}

class lda_blocks_t: public std::vector<lda_smat_t, callocator<lda_smat_t> >{ //{{{
	public:
		size_t nr_docs, nr_words, nr_blocks, start_doc, end_doc;
		std::vector<unsigned> block_of_doc;
		lda_blocks_t(){}

		// for multicore loading
		void load_data(size_t nr_blocks_, smat_t &wc) {
			auto range = [](size_t begin, size_t end)->std::vector<unsigned>{
				std::vector<unsigned> ret(end-begin);
				for(size_t i = 0; i < ret.size(); i++) ret[i] = begin+i;
				return ret;
			};

			nr_docs = wc.rows; nr_words = wc.cols; nr_blocks = nr_blocks_;
			start_doc = 0; end_doc = nr_docs;
			this->resize(nr_blocks);
			block_of_doc.resize(nr_docs);
			auto blocksize = nr_docs/nr_blocks + ((nr_docs%nr_blocks)?1:0); 
			for(auto t = 0U; t < nr_blocks; t++) {
				const std::vector<unsigned> &subset = range(t*blocksize, std::min((t+1)*blocksize, nr_docs));
				this->at(t).load_data(wc.row_subset_it(subset));
				for(auto it:subset) block_of_doc[it] = t;
			}
		}

		// for distributed loading 
		void load_data(int procid, petsc_reader_t &reader, doc_partitioner_t &partitioner) {
			nr_docs = reader.rows;
			nr_words = reader.cols;
			nr_blocks = partitioner.nr_samplers;
			start_doc = partitioner.start_doc(procid, 0);
			end_doc = partitioner.end_doc(procid, nr_blocks-1);
			this->resize(nr_blocks);
			for(auto samplerid = 0U; samplerid < nr_blocks; samplerid++) {
				size_t start_doc = partitioner.start_doc(procid, samplerid);
				size_t end_doc = partitioner.end_doc(procid, samplerid);
				this->at(samplerid).load_data(reader, start_doc, end_doc);
			}
		}
}; // }}}


template <typename val_type>
class model_vec_t: public std::vector<val_type, callocator<val_type>>{ // {{{
	public:
		model_vec_t(size_t size=0) {this->resize(size);}
		model_vec_t(size_t size, val_type &val) {this->resize(size, val);}
		int where; // the position in sampler_perm
		std::vector<int, callocator<int> > sampler_perm;
		void init_perm(int samplers) {
			where = 0;
			return; // XXX no permutation at all
			if(samplers > 0 and sampler_perm.size() == 0){
				sampler_perm.resize(samplers);
				for(auto i = 0; i < samplers; i++)
					sampler_perm[i] = i;
			}
		}
		void shuffle(unsigned *seed) { 
			auto nr_samplers = sampler_perm.size();
			for(size_t i = 0U; i < nr_samplers; i++) {
				size_t j = rand_r(seed) % (nr_samplers-i) + i;
				std::swap(sampler_perm[i], sampler_perm[j]);
			}
			//shuffle(sampler_perm.begin(), sampler_perm.end(), rng); 
		}
		int get_sampler() {
			if (where < (int)sampler_perm.size())
				return sampler_perm[where];
			else 
				return sampler_perm[where%(int)sampler_perm.size()];
		}
}; // }}}

template<typename val_type>
struct model_mat_t{ //{{{
	typedef model_vec_t<val_type> vec_t;
	size_t rows, cols, start_row, end_row;
	std::vector<vec_t, sallocator<vec_t> > buf;
	model_mat_t(size_t rows_=0, size_t cols_=0, size_t start_row_=0, size_t end_row_=0, int samplers=1) {
		resize(rows_, cols_, start_row_, end_row_, samplers);
	}
	vec_t& operator[](size_t id) {return buf[id-start_row];}
	const vec_t& operator[](size_t id) const {return buf[id-start_row];}
	void resize(size_t rows_=0, size_t cols_=0, size_t start_row_=0, size_t end_row_=0, int samplers=1) {
		if(end_row_ <= start_row_) end_row_ = rows_;
		rows = rows_; cols = cols_; start_row = start_row_; end_row = end_row_;
		buf.clear();
		buf.resize(end_row-start_row, vec_t(cols));
		if(buf.size() != end_row-start_row) {
			printf("No enough memory for model\n");
		}
		if(samplers > 1) 
			for(auto &vec: buf) 
				vec.init_perm(samplers);
	}
	void push_back(const vec_t &val, int samplers = 0) {
		buf.push_back(val); 
		buf[buf.size()-1].init_perm(samplers);
		end_row++;
	}

}; //}}}

// LDA Model {{{
//typedef std::vector<var_t, callocator<var_t> > vec_t;
//typedef std::vector<vec_t, sallocator<vec_t> > mat_t;
typedef model_vec_t<var_t> vec_t;
typedef model_mat_t<var_t> mat_t;
typedef std::vector<atomic<var_t>, callocator<atomic<var_t> > > atom_vec_t;
typedef std::vector<double, callocator<double>> double_vec_t;
typedef std::vector<double_vec_t, callocator<double_vec_t>> double_mat_t;

struct lda_model_t {
	size_t dim; // #topics
	size_t nr_docs, nr_words;
	mat_t Ndt, Nwt;
	vec_t Nt;

	lda_model_t(size_t dim_=0, size_t nr_docs_ =0, size_t nr_words_=0) {
		dim=dim_;nr_docs=nr_docs_;nr_words=nr_words_;
		Ndt.resize(nr_docs,dim);
		Nwt.resize(nr_words,dim);
		Nt.resize(dim);
	}
	lda_model_t(dist_lda_param_t &param, lda_blocks_t &blocks, int procid = 0) {
		dim = param.k; nr_docs = blocks.nr_docs; nr_words = blocks.nr_words;
		Nwt.resize(nr_words, dim, 0, nr_words, blocks.nr_blocks);
		Ndt.resize(nr_docs, dim, blocks.start_doc, blocks.end_doc);
		Nt.resize(dim);
	}

	void check(int signal) {
		for(auto i = 0U; i < nr_words; i++)
			for(auto t = 0U; t < dim; t++)
				if(Nwt[i][t] < 0) {
					printf("Wrong at %d\n", signal);
					return;
				}
	}
	void initialize_with_blocks(lda_blocks_t &blocks) {
		for(auto &block: blocks) 
			initialize_with_blocks(block);
	}

	void initialize_with_blocks(lda_smat_t &block) {
		for(size_t word = 0U; word < block.nr_words; word++) {
			for(auto idx = block.word_ptr[word]; idx != block.word_ptr[word+1]; idx++) {
				auto doc = block.doc_idx[idx];
				for(auto zidx = block.Z_ptr[idx]; zidx != block.Z_ptr[idx+1]; zidx++) {
					int topic = block.Zval[zidx];
					Ndt[doc][topic]++;
					Nwt[word][topic]++;
					Nt[topic]++;
				}
			}
		}
	}
}; // }}} 
/*
template <typename val_type>
struct token_pool_t{ // {{{
	typedef model_vec_t<val_type> token_t;
	int dim, nr_samplers;
	model_mat_t<val_type> pool;
	con_queue<int> avail_queue;
	token_pool_t(int dim, int nr_samplers, size_t init_size = 1024) {
		dim = dim; nr_samplers = nr_samplers;
		//pool.resize(init_size, token_t<val_type>(dim, nr_samplers));
		pool.resize(init_size, dim, 0, init_size, nr_samplers);
		for(auto pool_id = 0; pool_id < pool.size(); pool_id++) 
			avail_queue.push(pool_id);
	}
	token_t& operator[](int32_t pool_id) {return pool[pool_id];}
	const token_t& operator[](int32_t pool_id) const {return pool[pool_id];}
	size_t size() {return avail_queue.unsafe_size();}
	void push(int32_t pool_id) { avail_queue.push(pool_id); }
	int32_t pop() {
		int32_t pool_id = -1;
		bool succeed = avail_queue.try_pop(pool_id);
		if(succeed) return pool_id;
		else {
			pool_id = (int) pool.size();
			pool.push_back(token_t(dim), nr_samplers);
			return pool_id;
		}
	}
}; //}}}
*/

//print Nt, Nwt, Ndt count matrix
template <class type1 , class type2 , class type3>  void print_parameters(type1 Nt, type2 Nwt, type3 Ndt, const char* sampler_id)
{
    FILE *Nt_file; 
    char *cp = new char[1000];
    strcpy(cp, sampler_id);
    strcat(cp, ".Nt.txt");
    Nt_file = fopen(cp, "w+");
    for(auto i = Nt.begin();i!=Nt.end();++i) 
        fprintf(Nt_file, "%d\t", *i); 
    fclose(Nt_file);

    FILE *Nwt_file; 
    cp = new char[1000];
    strcpy(cp, sampler_id);
    strcat(cp, ".Nwt.txt");
    Nwt_file = fopen(cp, "w+");
    for(auto i = Nwt.begin();i!=Nwt.end();++i) {
        for(auto j=i->begin();j!=i->end();++j) 
            fprintf(Nwt_file, "%d\t", *j);
        fprintf(Nwt_file, "\n");
    }
    fclose(Nwt_file);

    FILE *Ndt_file; 
    cp = new char[1000];
    strcpy(cp, sampler_id);
    strcat(cp, ".Ndt.txt");
    Ndt_file = fopen(cp, "w+");
    for(auto i = Ndt.begin();i!=Ndt.end();++i) {
        for(auto j=i->begin();j!=i->end();++j)
            fprintf(Ndt_file, "%d\t", *j);
        fprintf(Ndt_file, "\n");
    }
    fclose(Ndt_file);
}

void print_doc_word_topic_triple(lda_smat_t &training, char* sampler_id) {
    FILE *token_file; 
    char *cp = new char[1000];
    strcpy(cp, sampler_id);
    strcat(cp, ".token.txt");
    token_file = fopen(cp, "w+");

    fprintf(token_file, "document word topic\n");
    for(auto d = 0U; d < training.nr_docs; d++) {
        for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
            auto w = training.word_idx[idx];
            for(auto Zidx = training.Z_ptr[idx]; Zidx != training.Z_ptr[idx+1]; Zidx++) {
                size_t topic = training.Zval[Zidx];
                fprintf(token_file, "%lu %lu %lu\n", d, w, topic);
            }
        }
    }
    fclose(token_file);
}

// take a lda_smat_t as input, return the d->w->(start_index, end_index) of topic storage map
std::map<size_t, std::map<size_t, std::pair<size_t,size_t> > > get_word_groups(lda_smat_t &training, int k){
    std::map<size_t, std::map<size_t, std::pair<size_t, size_t> > > groups;
    //printf("there are %lu lines \n",training.nnz);

    //training.initialize_Z_docwise(k);
    for(auto d = 0U; d < training.nr_docs; d++) {
        //printf("d:%lu \n" ,d);
        std::map<size_t,std::pair<size_t,size_t>> word_groups;
        for(auto idx = training.doc_ptr[d]; idx != training.doc_ptr[d+1]; idx++) {
            std::pair<size_t, size_t> st (training.Z_ptr[idx], training.Z_ptr[idx+1]-1);
            word_groups[training.word_count.col_idx[idx]]=st;
#ifdef DEBUG_GROUPS
            //if(d==0) 
            printf("z_idx %lu, doc %lu, word %lu, length %lu, start %lu, end %lu\n",
                    idx, d, training.word_count.col_idx[idx], training.Z_ptr[idx+1]-training.Z_ptr[idx], training.Z_ptr[idx], training.Z_ptr[idx+1]-1);
#endif
        }
        //printf("the doc %lu has size %lu \n",d,word_groups.size());
        groups[d] = word_groups;
    }
    return groups;
}

std::vector<size_t> calculate_exp_partition_groups(double ratio) {
    std::vector<size_t> groups;

    return groups;
}

template <class type1 , class type2 , class type3> void calculate_epsilon(lda_smat_t &training, lda_smat_t &test, FILE *log_file, 
        type1 Nt, type2 Nwt, type3 Ndt,
        int k, int iters, double alpha, double beta, double ratio) {
    double max_eps = 0, min_eps = 1000, total_eps = 0;
    size_t token_count = 0;

    double alphabar = alpha*k;
    double betabar = beta*training.nr_words;

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
                //-------------START draw sample for group
                std::vector<double> p(k);
                std::vector<double> p_group(k);
                double sum_p = 0, sum_p_group = 0;

                size_t old_z = training.Zval[group_s];

                //TODO calculate max log - min log value here

                //-------calculate P for group gibbs sampling 
                Nw[old_z] -= group_count; Nd[old_z] -= group_count; Nt[old_z] -= group_count;//decreament count

                for(auto t = 0; t < k; t++) { 
                    p_group[t] = ( (alpha+Nd[t])*(beta+Nw[t]) / (betabar+Nt[t]) );
                    sum_p_group += p_group[t];
                }
                for(auto t = 0; t < k; t++) {
                    p_group[t] = p_group[t]/sum_p_group;
                } 
                // printf("p_group = ");
                // for(auto t = 0; t < k; t++) {
                //     printf("%.3f ", p_group[t]);
                // }
                // printf("\n");
                Nw[old_z] += group_count; Nd[old_z] += group_count; Nt[old_z] += group_count;//increament count
                //-------END calculate P for group gibbs sampling 

                //-------calculate P for normal gibbs sampling 
                Nw[old_z]--; Nd[old_z]--; Nt[old_z]--;//decreament count

                for(auto t = 0; t < k; t++) { 
                    p[t] = ( (alpha+Nd[t])*(beta+Nw[t]) / (betabar+Nt[t]) );
                    sum_p += p[t];
                }
                for(auto t = 0; t < k; t++) {
                    p[t] = p[t]/sum_p;
                } 
                // printf("p=     ");
                // for(auto t = 0; t < k; t++) {
                //     printf("%.3f ", p[t]);
                // }
                // printf("\n");
                Nw[old_z]++; Nd[old_z]++; Nt[old_z]++;//increament count
                //-------END calculate P for normal gibbs sampling 


                double eps = 0;
                for(auto t = 0; t < k; t++) {
                    eps += std::abs(p[t]-p_group[t]);
                }
                max_eps = std::max(max_eps, eps);
                min_eps = std::min(min_eps, eps);
                total_eps += eps*group_count;
                token_count += group_count;

                //------------calcualte the next group start, end index
                sampled_count += (group_e-group_s+1);

                group_s = group_e+1; 
                if (group_count==0 || group_count+sampled_count>total_count) {
                    group_count = idx_e-group_s+1; //treat all of what remains as one group
                    //group_count=1;  //treat each of what remains as one group
                }
                group_e = group_s+group_count-1;
            }
        }
    }

    fprintf(log_file, "avg. epsilon=%.8f, max epsilon=%.8f, min epsilon=%.8f\n", total_eps/token_count, max_eps, min_eps);
    //fflush(stdout);
}

#endif // SPLDA_H

// nonzero entry for spvec_t
template<class T>
struct entry_t{ // {{{
    unsigned idx;
    T value;
    entry_t(unsigned idx_=0, T value_=T()): idx(idx_), value(value_){}
    bool operator<(const entry_t& other) const { return (idx < other.idx);}
    bool operator==(const entry_t& other) const {return idx == other.idx;}
}; // }}}
template<class T>
struct spidx_t: public std::vector<T>{ // {{{
    typedef typename std::vector<T> super;
    typedef size_t idx_t;
    std::vector<idx_t> nz_idx;
    spidx_t(size_t size_=0): super(size_) {}
    void gen_nz_idx() {
        nz_idx.clear();
        for(auto t = 0U; t < this->size(); t++) 
            if((*this)[t] > 0) nz_idx.push_back((idx_t)t);
    }
    void push(idx_t idx) {
        size_t len = nz_idx.size();
        if(len)  {
            nz_idx.push_back(idx); // nz_idx.size() = len+1 now.
            idx_t *start = nz_idx.data(), *end=start+len;
            idx_t *ptr = std::lower_bound(start, end, idx);
            if(ptr != end) {
                memmove(ptr+1, ptr, (len - (ptr-start))*sizeof(idx_t));
                *ptr = idx;
                push_call_count++;
            }
        } else {
            nz_idx.push_back(idx);
        }
    }
    void pop(idx_t idx) {
        idx_t *start = nz_idx.data(), *end = start+nz_idx.size();
        idx_t *ptr = std::lower_bound(start, end, idx);
        if(ptr != end) remove_ptr(ptr);
    }
    void remove_ptr(idx_t *ptr) {
        size_t cnt = nz_idx.size()-(ptr+1-nz_idx.data());
        if(cnt) {
            memmove(ptr, ptr+1, cnt*sizeof(idx_t));
            remove_call_count++;
        }
        nz_idx.pop_back();
    }
}; //}}}
struct alias_table { // {{{
    size_t size;
    std::vector<size_t> alias;
    std::vector<double> prob;
    std::vector<double> P;
    std::vector<size_t> L,S;
    alias_table(size_t size_ = 0, const double *p = NULL): size(size_){build_table(size_, p);}
    alias_table(std::vector<double>& p): alias_table(p.size(), p.data()) {}
    void build_table(size_t size_, const double *p) {
        size = size_;
        if(size) {
            alias.resize(size);
            prob.resize(size);
            P.resize(size);
            L.resize(size);
            S.resize(size);
        } else return;
        if(p==NULL) return;
        //std::vector<double>P(size);
        double sum = 0.0;
        for(size_t i = 0; i < size; ++i) sum+=p[i];
        for(size_t i = 0; i < size; ++i) P[i] = p[i]*size/sum;
        //std::vector<size_t>L(size), S(size);
        int nS = 0, nL = 0;
        for(size_t i = 0; i < size; ++i) {
            if(P[i] < 1) S[nS++] = i;
            else L[nL++] = i;
        }
        while( nS && nL ) {
            size_t a = S[--nS]; // Schwarz's l
            size_t g = L[--nL]; // Schwarz's g
            prob[a] = P[a];
            alias[a] = g;
            P[g] = P[g] + P[a] - 1;
            if ( P[g] < 1 ) S[nS++] = g;
            else L[nL++] = g;
        }
        while(nL) prob[L[--nL]] = 1;
        while(nS) prob[S[--nS]] = 1;
    }
    void build_table(std::vector<double>& p) { build_table(p.size(), p.data()); }
    size_t sample(double urnd1, double urnd2) {
        size_t idx = (size_t)floor(urnd1*size);
        return urnd2 < prob[idx] ? idx : alias[idx];
    }
    size_t sample(double urnd2) {
        size_t idx = rand()%size;
        return urnd2 < prob[idx] ? idx : alias[idx];
    }
}; // }}}
// F+Tree Implementation
struct htree_t{ // {{{
    size_t size;     // real size of valid 
    size_t elements; // 2^ceil(log2(size))
    std::vector<double> val;
    double *true_val;
    size_t init_time, update_time, sample_time;
    double& operator[](size_t idx) { assert(idx < elements); return true_val[idx]; }
    const double& operator[] (size_t idx) const { assert(idx < elements); return true_val[idx]; }
    double get_init_time() {return (double)init_time/((CLOCKS_PER_SEC/1000)/1000);}
    double get_update_time() {return (double)update_time/((CLOCKS_PER_SEC/1000)/1000);}
    double get_sample_time() {return (double)sample_time/((CLOCKS_PER_SEC/1000)/1000);}
    void init_dense(){
        for(size_t pos = elements-1; pos > 0; --pos) 
            val[pos] = val[pos<<1] + val[(pos<<1)+1];
    }
    void update_parent(size_t idx, double delta) { // {{{
        idx = (idx+elements)>>1;
        while(idx) {
            val[idx] += delta;
            idx >>= 1;
        }
    } // }}}
    void update(size_t idx) { // {{{
        double value = val[idx+=elements]; // delta update
        value -= (val[idx>>1]-val[idx^1]);
        while(idx) {
            val[idx] += value;
            idx >>= 1;
        }
    } // }}}
    void set_value(size_t idx, double value) { // {{{
        value -= val[idx+=elements]; // delta update
        while(idx) {
            val[idx] += value;
            idx >>= 1;
        }
    } // }}}
    // urnd: uniformly random number between [0,1]
    size_t log_sample(double urnd) { // {{{
        //urnd *= val[1]; 
        size_t pos = 1;
        while(pos < elements) {
            pos <<= 1; 
            if(urnd >= val[pos]) 
                urnd -= val[pos++];
            /*
               double tmp = urnd - val[pos];
               if(tmp >= 0) {
               urnd = tmp;
               pos++;
               }*/
            /*
               if(urnd < val[pos*2]) 
               pos = pos*2;
               else { 
               urnd -= val[pos*2]; 
               pos = pos*2+1; 
               }
               */
        }
        return pos-elements;
    } // }}}
    size_t linear_sample(double urnd) { // {{{
        urnd = urnd*val[1];
        size_t pos = elements;
        while(urnd > 0)
            urnd -= val[pos++];
        return pos-elements;
    } // }}}
    double total_sum() {return val[1];}
    htree_t(size_t size=0): init_time(0),update_time(0),sample_time(0) { resize(size); }
    void resize(size_t size_ = 0) { // {{{
        size = size_;
        if(size == 0) val.resize(0);
        elements = 1;
        while(elements < size) elements *= 2;
        val.clear(); val.resize(2*elements, 0.0);
        true_val = &val[elements];
    } //}}}
    void clear() { for(auto &v: val) v = 0; }
}; // }}}

struct sample_pool_t{ //  {{{
    std::vector<unsigned> pool;
    std::vector<double> qw;
    double Qw;
    size_t capacity;
    sample_pool_t(size_t capacity=0): capacity(capacity){
        pool.reserve(capacity); qw.reserve(capacity); Qw=0;
    }
    void reserve(size_t capacity=0){pool.reserve(capacity); qw.resize(capacity);}
    bool next_sample(unsigned &sample) {
        if(pool.empty()) return false;
        sample = pool.back(); pool.pop_back();
        return true;
    }
    unsigned next_sample() {
        double sample = pool.back(); pool.pop_back();
        return sample;
    }
    bool empty() {return pool.empty();}
    void refill(alias_table& A) {
        A.build_table(qw);
        Qw = 0; for(auto &v: qw) Qw += v;
        pool.clear();
        while(pool.size() < capacity)
            pool.push_back(A.sample(drand48(),drand48()));
    }
}; // }}}

