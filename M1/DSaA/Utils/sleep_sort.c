#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/wait.h>

#define True    1
#define False   0

// Sleep Sort Algorithm

void swap(int* arr, int src, int dst){
    int tmp = arr[src];
    arr[src] = arr[dst];
    arr[dst] = tmp;
}

void sleep_sort(int* arr, unsigned size){
    // struct timespec ts1, ts2;
    while(size){
        if(!fork()){ // child
            // ts1.tv_sec = 0;
            // ts1.tv_nsec = arr[size-1];
            // nanosleep(&ts1, &ts2);
            sleep(arr[size-1]);
            printf("%d\n", arr[size-1]);
            break;
        }   --size;
    }
}

// =====================

int main(){

    int arr[] = {3, 8, 13, 4, 5, 2, 2, 7, 3};
    size_t size = sizeof(arr)/sizeof(int);

    for(size_t i=0; i<size; ++i){
        printf("%d ", arr[i]);
    }   printf("\n");

    sleep_sort(arr, size);

    return 0;
}