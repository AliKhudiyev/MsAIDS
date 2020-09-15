#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define True    1
#define False   0

// Merge Sort Algorithm

void swap(int* arr, int src, int dst){
    int tmp = arr[src];
    arr[src] = arr[dst];
    arr[dst] = tmp;
}

void copy(int* dst, int* src, int beg, int end){
    for(int i=0; beg<end; ++beg)
        dst[i++] = src[beg];
}

int* merge(int* arr1, int size1, int* arr2, int size2){
    int* arr = (int*)malloc((size1+size2) * sizeof(int));
    int i = 0, j = 0;

    for(int p=0; p<size1+size2; ++p){
        if(j>=size2 || (i<size1 && arr1[i] < arr2[j])){
            arr[p] = arr1[i];
            ++i;
        } else{
            arr[p] = arr2[j];
            ++j;
        }
    }

    return arr;
}

void merge_sort(int* arr, unsigned size){
    if(size>1){
        int* arr1 = (int*)malloc(size/2 * sizeof(int));
        int* arr2 = (int*)malloc((size-size/2) * sizeof(int));
        
        copy(arr1, arr, 0, size/2);
        copy(arr2, arr, size/2, size);

        merge_sort(arr1, size/2);
        merge_sort(arr2, size-size/2);

        int* res = merge(arr1, size/2, arr2, size-size/2);
        copy(arr, res, 0, size);

        free((void*)arr1);
        free((void*)arr2);
        free((void*)res);
    }
}

// =====================

int main(){

    int arr[] = {3, 8, 13, 4, 5, 2, 2, 7, 3};
    size_t size = sizeof(arr)/sizeof(int);

    for(size_t i=0; i<size; ++i){
        printf("%d ", arr[i]);
    }   printf("\n");

    merge_sort(arr, size);

    for(size_t i=0; i<size; ++i){
        printf("%d ", arr[i]);
    }   printf("\n");

    return 0;
}