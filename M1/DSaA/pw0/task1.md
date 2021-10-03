# Exercise 1

which - to locate programs in user's path

whereis - to locate programs in disk

apropos - search the whatis database for strings

mkdir - to create directories

cp - to copy

rm - to remove files

# Exercise 2

mkdir pw01 - pw01 directory is created

cd pw01 - changes current directory to pw01

pwd - prints working directory

touch fileA - creates a file named fileA

touch fileB fileC fileD - creates 3 files named fileB, fileC and fileD

touch 02{4,7,8,9}.vi - creates files names 024.vi, 027.vi and so on.

ls - lists the content of current directory(not fully)

mkdir docs - creates a directoru called docs

mkdir -p docs/dirU/{dirUA, dirUB, dirUC/dirUCA} - creates those directories recusively

cd docs/dirU/dirUC/dirUCA - changes directory to docs/dirU/dirUC/dirUCA

pwd - ?

touch progUCA{}.exe - ?

ls -al - ?

pwd - ?

cd ../../../.. - ?

# Exercise 3

find docs -name '*9*' -exec rm {} \;

find docs -name '*.ttt' -exec rm {} \;

chmod 

find docs -name 'dirUB' -exec rm -r {} \;

# Exercise 4

1. echo Today is $(date)

2. echo Today is $(date +%d/%m/%Y)

# Exercise 5

1. head -n 10 EX5
2. head -n 15 EX5
3. tail -n 20 EX5
4. less EX5
5. line/words/bytes of EX5

# Exercise 6

1. sort EX6
2. sort EX6 | uniq -u
3. sort EX6 | uniq -u > newEX6

# Exercise 7

1. tar xvf EX7.tar.gz
2. ls -R EX7
3. find EX7 -exec file {} \;
