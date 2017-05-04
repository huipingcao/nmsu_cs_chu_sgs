#include <map>
#include <set>
#include <algorithm>
#include <cstdio>
using namespace std;

bool comp_fun(pair<int,int> a, pair<int,int> b){
    return a.second < b.second;
}

int main() {

    map<int, int> kv_map;
    kv_map[1] = 10;
    kv_map[2] = 5;
    kv_map[3] = 6;
    kv_map[4] = 2;
    kv_map[5] = 1;
    kv_map[6] = 2;
    kv_map[7] = 4;

    map<int, int>::iterator max_ele = max_element(kv_map.begin(), kv_map.end(), comp_fun);
    printf("max k,v = %d, %d\n", max_ele->first, max_ele->second);

    set<int> set1;
    set<int> set2;
    set2.insert(1);
    set2.insert(2);
    set2.insert(3);
    set2.insert(4);
    set2.insert(5);
    set2.insert(6);
    set1.insert(set2.begin(), set2.end());
    printf("set1.size=%d", set1.size());

}
