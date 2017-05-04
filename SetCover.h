#include <cstdio>
#include <cstdlib>
#include <cstring>
//#include <algorithm>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <assert.h>
#include <iostream>

template<class T> void print_vec(const std::vector<T> vec)
{
    for(T x:vec)
    {
        std::cout<<x.first<<"        "<<x.second<<'\n';
    }
}

bool pairCompByValue(std::pair<size_t,size_t> firstElm,
        std::pair<size_t,size_t> secondElm)
{
    return firstElm.second < secondElm.second;
}


void putTokensToCovered(std::set<size_t> &covered, 
        std::set<size_t> &uncovered, 
        std::pair<size_t,size_t> c_doc_idx,
        lda_smat_t &training) {
    for(int w_idx=training.doc_ptr[c_doc_idx.first]; w_idx != training.doc_ptr[c_doc_idx.first+1]; w_idx++){
        size_t token =training.word_idx[w_idx];
        covered.insert(token);
        auto it = uncovered.find(token);
        if (it!=uncovered.end())
            uncovered.erase(it);
    } 
}

void updateInfo(std::set<size_t> &uncovered, 
        size_t d,
        std::vector< std::pair<size_t,size_t> > &c_docs,
        std::map<size_t, std::set<size_t> > &w_d_map,
        std::map<size_t, std::set<size_t> > &d_w_map, 
        lda_smat_t &training) {
    /* for(int i = 0 ; i < c_docs.size();) {
       size_t count=0;
       for(size_t w_idx=training.doc_ptr[c_docs[i].first]; w_idx != training.doc_ptr[c_docs[i].first+1]; w_idx++) {
       size_t token = training.word_idx[w_idx];
       if(uncovered.find(token) != uncovered.end())
       count++;
       }
       if(count==0)
       c_docs.erase(c_docs.begin()+i);
       else {
       c_docs[i].second=count;
       i++;
       }
       } */
    std::set<size_t> covered_w_set = d_w_map[d];
    std::set<size_t> updated_d_set;
    for (std::set<size_t>::iterator w_it=covered_w_set.begin(); w_it!=covered_w_set.end(); w_it++) {
        std::set<size_t> d_set = w_d_map[*w_it];
        updated_d_set.insert(d_set.begin(), d_set.end());
    }
    
   // printf("updated_d_set.size=%d, covered_w_set.size=%d, ", updated_d_set.size(), covered_w_set.size());
    for (std::set<size_t>::iterator d_it=updated_d_set.begin(); d_it!=updated_d_set.end(); d_it++) {
        std::set<size_t> *w_set = &d_w_map[*d_it];
        for (std::set<size_t>::iterator w_it=w_set->begin(); w_it!=w_set->end(); ) {
            if (covered_w_set.find(*w_it)!=covered_w_set.end())
                w_it = w_set->erase(w_it);
            else w_it++;
        }
        if (w_set->size()==0)
            d_w_map.erase(*d_it);
    }
    for (std::set<size_t>::iterator w_it=covered_w_set.begin(); w_it!=covered_w_set.end(); w_it++) {
        std::set<size_t> *d_set = &w_d_map[*w_it];
        for (std::set<size_t>::iterator d_it=d_set->begin(); d_it!=d_set->end(); ) {
            if (updated_d_set.find(*d_it)!=updated_d_set.end())
                d_it = d_set->erase(d_it);
            else d_it++;
        }
        if (d_set->size()==0)
            w_d_map.erase(*w_it);
    }

    // END update 

    for(int i = 0 ; i < c_docs.size();) {
        if (updated_d_set.find(c_docs[i].first)==updated_d_set.end()) {i++; continue; } //skip unnecessary documents

        size_t count=0;
        for(size_t w_idx=training.doc_ptr[c_docs[i].first]; w_idx != training.doc_ptr[c_docs[i].first+1]; w_idx++) {
            size_t token = training.word_idx[w_idx];
            if(uncovered.find(token) != uncovered.end())
                count++;
        }
        if(count==0)
            c_docs.erase(c_docs.begin()+i);
        else {
            c_docs[i].second=count;
            i++;
        }
    }
}

std::set<size_t> getMinimalDataSet(lda_smat_t &training) {
    std::set<size_t> uncovered;
    std::set<size_t> covered;
    std::set<size_t> min_docs;
    std::vector< std::pair<size_t,size_t> > d_w_vec;
    std::map<size_t, std::set<size_t> > w_d_map;
    std::map<size_t, std::set<size_t> > d_w_map; 

    for(int i = 0; i<training.word_count.nnz;i++) {
        size_t token = training.word_idx[i];
        uncovered.insert(token);
        std::set<size_t> d_set;
        w_d_map[token] = d_set;
    }

    for(int d = 0 ; d<training.nr_docs; d++) {
        std::set<size_t> w_set;
        for(size_t w_idx=training.doc_ptr[d]; w_idx != training.doc_ptr[d+1]; w_idx++) {
            size_t token = training.word_idx[w_idx];
            w_d_map[token].insert(d);
            w_set.insert(token);
        }
        d_w_map[d] = w_set;

        std::pair<size_t,size_t> st (d, w_set.size());
        d_w_vec.push_back(st);
    }

    while( uncovered.size()!= 0 ) {
        // printf("uncovered left=%d\n", uncovered.size());

        size_t starttime = std::clock();
        std::vector<std::pair<size_t, size_t> >::iterator max_cover_pair = 
            std::max_element(d_w_vec.begin(), d_w_vec.end(), pairCompByValue);
        size_t endtime = std::clock();
        // printf("find max time=%.2f, ", (double)(endtime-starttime)/(CLOCKS_PER_SEC/1000));
        printf("d=%d, covered_w=%d \n", max_cover_pair->first, max_cover_pair->second);

        starttime = std::clock();
        putTokensToCovered(covered, uncovered, *max_cover_pair, training);
        endtime = std::clock();
        // printf("update set time=%.2f, ", (double)(endtime-starttime)/(CLOCKS_PER_SEC/1000));

        min_docs.insert(max_cover_pair->first);
        // d_w_vec.erase(max_cover_pair);

        starttime = std::clock();
        updateInfo(uncovered, max_cover_pair->first, d_w_vec, w_d_map, d_w_map, training);
        endtime = std::clock();
        // printf("d=%d, covered_w=%d, ", max_cover_pair->first, max_cover_pair->second);
        // printf("update info time=%.2f \n", (double)(endtime-starttime)/(CLOCKS_PER_SEC/1000));
    }

    return min_docs;
}


std::set<size_t> getMinimalDataSetWithUncoverd(lda_smat_t &training,
        double ratio)
{
    std::set<size_t> resultSet = getMinimalDataSet(training);
    //printf("find mincover set %lu\n",resultSet.size());
    int needMore = (training.nr_docs-resultSet.size())*ratio;
    int counter = 0;
    for(int i = 0 ; i<training.nr_docs;i++)
    {
        if(counter == needMore)
        {
            break;
        }

        if(resultSet.find(i)==resultSet.end())
        {
            resultSet.insert(i);
            counter++;
        }
    }
    //printf("find mincover set %lu\n",resultSet.size());
    return resultSet;
}
