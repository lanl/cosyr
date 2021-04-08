# Gnuplot script for thread scaling

reset
set terminal postscript eps enhanced color 14 size 9cm,8cm
set output "thread_scaling.eps"
set size ratio 0.85
set xlabel "threads\n"
set ylabel "(s)"
set key below maxcols 1

set xrange [1:36]
set logscale x 2
#set logscale y 2

set title "thread strong scaling"
set grid
plot 'thread_scaling.dat' u 1:8  t 'kernel' w lp lc rgb '#CB0707' pt 5,\
     'thread_scaling.dat' u 1:4  t 'remap'  w lp lc rgb '#800080' pt 2,\
     'thread_scaling.dat' u 1:11 t 'update'  w lp lc rgb '#4682B4' pt 2,\
     'thread_scaling.dat' u 1:12 t 'total' w lp lc rgb '#000000' pt 3
