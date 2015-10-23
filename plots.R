setwd('/home/ariel/Documents/ac/clusterCoeff/ccgen_test')
library('ggplot2')

data <- read.csv('data100_10.csv', sep = '|')
colnames(data) <- c('p', 'ecc', 'cc')

pdf('plot100_10.pdf')
p2 <- ggplot(data, aes(x=p, y=ecc)) +
  geom_line(alpha=.3) +
  geom_point(aes(cc, color=as.factor(cc))) +
  ggtitle("Clustering coeff. (real vs theoretical)") + 
  xlab('probability') +
  ylab('clustering coeff')

p2
dev.off()

