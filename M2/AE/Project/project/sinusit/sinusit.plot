set term png
set output "sinusit.png"
set xrange[0:2340000]
set xlabel "Number of Evaluations"
set ylabel "Fitness"
plot 'sinusit.dat' using 3:4 t 'Best Fitness' w lines, 'sinusit.dat' using 3:5 t  'Average' w lines, 'sinusit.dat' using 3:6 t 'StdDev' w lines
set term png
set output "sinusit.png"
set xrange[0:430000]
set xlabel "Number of Evaluations"
set ylabel "Fitness"
plot 'sinusit.dat' using 3:4 t 'Best Fitness' w lines, 'sinusit.dat' using 3:5 t  'Average' w lines, 'sinusit.dat' using 3:6 t 'StdDev' w lines
