#include "gpuvector.cuh"
#include "utils.cuh"
#include "perf.cuh"
#include "scan.cuh"
#include "operators.cuh"
#include "bulk.cuh"

#include <vector>
#include <iostream>

using namespace std;


int main()
{
    size_t const N(100);
    auto hX = std::vector<float>(N);
    for (int i = 0; i< N; i++){
    	hX[i] = i;
    }
    std::vector<int> hI = {1,12,13,14,14,18,20,38,39,44,45,50,50,50,54,56,59,63,68,69,74,75,84,84,88,111,111,119,121,123,126,127,144,153,157,159,163,169,169,175,178,183,190,194,195,196,196,201,219,219,253,256,259,262,262,266,272,273,278,283,284,291,296,297,302,303,306,306,317,318,318,319,319,320,320,323,326,329,330,334,340,349,352,363,366,367,369,374,381,383,383,384,386,388,388,389,393,398,398,399};

    auto dX = gpuVector<float>(hX);
    auto dI = gpuVector<int>(hI);
    auto P = findBalancedPartitions(dI, dX);
    cout<<P<<endl;
    return 0;
}
