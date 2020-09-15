#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define True    1
#define False   0

// Insertion Sort Algorithm

void swap(int* arr, int src, int dst){
    int tmp = arr[src];
    arr[src] = arr[dst];
    arr[dst] = tmp;
}

void insert(int* arr, int dst, int src){
    int s = -1;
    if(src < dst) s = 1;

    for(int i=src; i!=dst; i+=s){
        swap(arr, i, i+s);
    }
}

void insertion_sort(int* arr, unsigned size){
    int p=1;
    for(int i=0; i<size-1; ++i, ++p){
        if(arr[i+1] < arr[i]){
            for(int j=0; j<p; ++j){
                if (arr[i+1] <= arr[j]){
                    insert(arr, j, i+1);
                    break;
                }
            }
        }
    }
}

// =====================

int main(){

    int arr[] = {3, 8, 13, 4, 5, 2, 2, 7, 3};
    size_t size = sizeof(arr)/sizeof(int);

    for(size_t i=0; i<size; ++i){
        printf("%d ", arr[i]);
    }   printf("\n");

    insertion_sort(arr, size);

    for(size_t i=0; i<size; ++i){
        printf("%d ", arr[i]);
    }   printf("\n");

    return 0;
}