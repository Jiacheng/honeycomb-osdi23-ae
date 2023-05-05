#set nokey

set datafile separator comma
set mono
#set terminal lua tikz size 3in,1.8in font ",6"

set terminal pdf
set output "validation-time.pdf"

#set style data lines
set xrange[32:2048]
set yrange[0.02:327.68]
set logscale x 2
set logscale y 2
set ylabel "Validation Time (ms)" offset -1, 0
set xlabel "Number of instructions in the kernel" offset 0, -1
set xtics rotate by 30 right nomirror
set ytics nomirror
set xtics nomirror in 
set xtics 2
set mxtics 2
set ytics 4
set mytics 2
set format x "2^{%L}"
set format y "2^{%L}"

#set style fill pattern 1

plot "validation-time.csv" \
  using 1:2 with points notitle lc rgb "web-blue"

