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

CSV* read_csv(const char* filepath){
    CSV* csv = malloc(sizeof(CSV));

    // Initializing and allocating memory for CSV
    tell_shape(filepath, csv);

    printf("Shape: %u x %u\n", csv->nrow, csv->ncol);

    csv->context = malloc(csv->nrow * sizeof(double));
    for(unsigned r=0; r<csv->nrow; ++r){
        csv->context[r] = malloc(csv->ncol * sizeof(double));
        for(unsigned c=0; c<csv->ncol; ++c){
            csv->context[r][c] = malloc(MAX_CHARS/10);
        }
    }

    // Reading the file into CSV pointer
    FILE* file = fopen(filepath, "r");
    
    char line[MAX_CHARS];
    char *token, *ptr = line;
    unsigned row=0;

    while(fgets(line, MAX_CHARS, file)){
        ptr = line;
        for(unsigned col=0; col<csv->ncol; ++col){
            char* beg = ptr;
            char* end = strstr(beg, _csv_delim);

            if(beg==end){
                csv->context[row][col] = NULL_CELL;
            }
            else{
                token = strtok(beg, _csv_delim);
                // csv->context[row][col] = atof(token);
                strncpy(csv->context[row][col], token, MAX_CHARS/10);
                clear_str(csv->context[row][col], MAX_CHARS/10);
            }   ptr = end+1;
        }
        ++row;
    }

    fclose(file);
    
    return csv;
}

void write_csv(const char* filepath, const CSV* csv){
    FILE* file = fopen(filepath, "w");

    for(unsigned i=0; i<csv->nrow; ++i){
        for(unsigned j=0; j<csv->ncol; ++j){
            fprintf(file, "%s", csv->context[i][j]);
            if(j<csv->ncol-1){
                fprintf(file, "%s", _csv_delim);
            }
        }   fprintf(file, "%c", '\n');
    }

    fclose(file);
}

void print_csv(const CSV* csv){
    for(unsigned r=0; r<csv->nrow; ++r){
        for(unsigned c=0; c<csv->ncol; ++c){
            printf("%s", csv->context[r][c]);
            if(c<csv->ncol-1) printf(",\t");
        }   printf("\n");
    }
}

void free_csv(CSV* csv){
    for(unsigned i=0; i<csv->nrow; ++i) free((void*)csv->context[i]);
    free((void*)csv->context);
    free((void*)csv);
    csv = NULL;
}

void set_delim(char delim){
    _csv_delim[0] = delim;
}
