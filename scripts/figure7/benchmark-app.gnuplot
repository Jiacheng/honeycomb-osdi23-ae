#set nokey

set datafile separator comma
#set mono
set terminal lua tikz size 3.33in,1.8in font ",6"

set terminal pdf
set output "./benchmarks-app.csv"

set style data histograms
# set style histogram rowstacked
set boxwidth 0.75 relative

#set style data lines
#set xrange[0:1073741824]
set yrange[0:4]
set ylabel "Relative time"
set xtics rotate by 45 right nomirror
set ytics nomirror
set style fill solid 0.75 
set key top center horizontal width 5

plot ARGV[1] \
  using ($3/$2):xtic(1) title "Honeycomb runtime baseline", \
  "" using ($4/$2):xtic(1) title "Honeycomb runtime SM", \
  "" using ($5/$2):xtic(1) title "Honeycomb runtime SH+Mem", \
  "" using ($6/$2):xtic(1) title "Honeycomb runtime SH+Mem+V" \
