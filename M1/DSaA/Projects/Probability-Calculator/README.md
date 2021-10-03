# Probability Calculator

This project is a simple probability calculator which prints the probability accoring to given arguments. The arguments are shown below:

- Distribution type: Normal/Binomial/Chi Square/Student's T/F
- Z-value or (Row, Column) or (Arg, Row, Column) of the given probability distribution table

## How to run

Open the project directory in terminal and then type:

```bash
cd build; cmake .. && make
```

If _make_ complains about `#include <omp.h>` then make sure that you are compiling the project with _gcc_ and not ~~clang~~ and type in the terminal:

```bash
export CC=$(which gcc)
```

Now you can run the program by typing the following command:

```bash
./pc [distribution type]
```

For example,

```bash
./pc 1
Enter row and column arguments: 4 0.8
Probability: 0.270730
```

There are 5 supported distribution types and here are their "code numbers":

| Distribution type | Code number |
|--|--|
| Normal | 0 |
| Student's t | 1 |
| Chi square | 2 |
| Binomial | 3 |
| Fisher | 4 |

To generate *Student's t*, *Chi square* and *Fisher* distributions type `./pc -g`.
