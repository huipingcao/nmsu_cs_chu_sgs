#include <unordered_set>
#include <cstring>

int main(){
    std::unordered_set<size_t> min_set;
    FILE *min_set_file; 
    char *cp = new char[1000];
    strcat(cp, "/data/chu/lda-data/nomad_lda_data/enron/min_set.txt");
    min_set_file = fopen(cp, "r");

    size_t d = 0U;
    while(fscanf(min_set_file, "%u\n", &d)!=EOF) {
        min_set.insert(d);
    }
    fclose(min_set_file);

    for (std::unordered_set<size_t>::iterator it=min_set.begin(); it!=min_set.end(); it++) {
        // printf("%u \n", *it);
        // if (min_set.find(*it)!=min_set.end())
        //     printf("yest");
    }

    d = 411; //TODO  problem is here
    if (min_set.find(d)!=min_set.end())
        printf("yes");

    return 0;
}
