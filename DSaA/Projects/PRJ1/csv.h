
#ifndef _CSV_
#define _CSV_

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CHARS 4000

extern char _csv_delim[];

typedef struct{
    unsigned ncol, nrow;
    double** context;
}CSV;

void tell_shape(const char* filepath, CSV* csv);
void read_csv(const char* filepath, CSV* csv);
void print_csv(const CSV* csv);
void set_delim(char delim);

#endif
