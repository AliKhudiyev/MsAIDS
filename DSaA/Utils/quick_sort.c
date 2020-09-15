#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Quick Sort Algorithm

void swap(int* arr, int src, int dst){
    int tmp = arr[src];
    arr[src] = arr[dst];
    arr[dst] = tmp;
}

int partition(int* arr, int beg, int end){
    int i = beg+1, j = end-1;

    while(i<j){
        for(; arr[i] <= arr[beg] && i < end; ++i);
        for(; arr[j] > arr[beg] && j >= 0; --j);
        if(i<j) swap(arr, i, j);
    }   swap(arr, beg, j);

    return j;
}

void quick_sort_(int* arr, int beg, int end){
    if(beg<end){
        int pivot = partition(arr, beg, end);
        quick_sort_(arr, beg, pivot);
        quick_sort_(arr, pivot+1, end);
    }
}

void quick_sort(int* arr, unsigned size){
    quick_sort_(arr, 0, (int)size);
}

// =====================

int main(){

    int arr[] = {3, 8, 13, 4, 5, 2, 2, 7, 3};
    size_t size = sizeof(arr)/sizeof(int);

    for(size_t i=0; i<size; ++i){
        printf("%d ", arr[i]);
    }   printf("\n");

    quick_sort(arr, size);

    for(size_t i=0; i<size; ++i){
        printf("%d ", arr[i]);
    }   printf("\n");

    return 0;
}