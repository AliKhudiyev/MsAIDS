#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define True    1
#define False   0

// Selection Sort Algorithm

void swap(int* arr, int src, int dst){
    int tmp = arr[src];
    arr[src] = arr[dst];
    arr[dst] = tmp;
}

void selection_sort(int* arr, unsigned size){
    int i = 0, j = 0, p = 0;
    do{
        for(j=i; i<size; ++i){
            if(arr[j] > arr[i]){
                j = i;
            }
        }   swap(arr, p, j);
    }while((i=++p) < size);
}

// =====================

int main(){

    int arr[] = {3, 8, 13, 4, 5, 2, 2, 7, 3};
    size_t size = sizeof(arr)/sizeof(int);

    for(size_t i=0; i<size; ++i){
        printf("%d ", arr[i]);
    }   printf("\n");

    selection_sort(arr, size);

    for(size_t i=0; i<size; ++i){
        printf("%d ", arr[i]);
    }   printf("\n");

    return 0;
}