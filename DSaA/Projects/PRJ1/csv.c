#include "csv.h"

char _csv_delim[] = ",";

void tell_shape(const char* filepath, CSV* csv){
    csv->ncol = 1;
    csv->nrow = 0;

    char tmp;

    FILE* file = fopen(filepath, "r");

    while((tmp=fgetc(file)) != '\n'){
        if(tmp == _csv_delim[0]) ++csv->ncol;
    }

    fseek(file, 0, 0);
    char line[MAX_CHARS];
    while(fgets(line, MAX_CHARS, file)){
        ++csv->nrow;
    }

    fclose(file);
}

void read_csv(const char* filepath, CSV* csv){
    // Initializing and allocating memory for CSV
    tell_shape(filepath, csv);

    printf("Shape: %u x %u\n", csv->nrow, csv->ncol);

    csv->context = malloc(csv->nrow * sizeof(double));
    for(unsigned r=0; r<csv->nrow; ++r){
        csv->context[r] = malloc(csv->ncol * sizeof(double));
    }

    // Reading the file into CSV pointer
    FILE* file = fopen(filepath, "r");
    
    char line[MAX_CHARS];
    char *token, *ptr = line;
    unsigned row=0, col=0;

    while(fgets(line, MAX_CHARS, file)){
        ptr = line;
        if(line[0] == _csv_delim[0]) ++col;
        while((token = strtok(ptr, _csv_delim))){
            csv->context[row][col++] = atof(token);
            ptr = NULL;
        }   ++row; col = 0;
    }

    fclose(file);
}

void print_csv(const CSV* csv){
    for(unsigned r=0; r<csv->nrow; ++r){
        for(unsigned c=0; c<csv->ncol; ++c){
            printf("%lf, ", csv->context[r][c]);
        }   printf("\n");
    }
}

void set_delim(char delim){
    _csv_delim[0] = delim;
}
