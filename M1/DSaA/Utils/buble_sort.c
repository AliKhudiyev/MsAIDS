#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define True    1
#define False   0

// Bubble Sort Algorithm

void swap(int* arr, int src, int dst){
    int tmp = arr[src];
    arr[src] = arr[dst];
    arr[dst] = tmp;
}

void buble_sort(int* arr, unsigned size){
    int pass;

    do{
        pass = False;
        for(unsigned i=0; i<size-1; ++i){
            if(arr[i] > arr[i+1]){
                swap(arr, i, i+1);
                pass = True;
            }
        }
    }while(pass);
}

// =====================

int main(){

    int arr[] = {3, 8, 13, 4, 5, 2, 2, 7, 3};
    size_t size = sizeof(arr)/sizeof(int);

    for(size_t i=0; i<size; ++i){
        printf("%d ", arr[i]);
    }   printf("\n");

    buble_sort(arr, size);

    for(size_t i=0; i<size; ++i){
        printf("%d ", arr[i]);
    }   printf("\n");

    return 0;
}