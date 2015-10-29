setwd('/home/ariel/Documents/ac/clusterCoeff/ccgen_test')
library('ggplot2')

data <- read.csv('./data/data50_100_t.csv', sep = '|')
colnames(data) <- c('p', 'ecc', 'cc')

pdf('./results/plot50_100.pdf')
p1 <- ggplot(data, aes(x=p, y=ecc)) +
  geom_line(alpha=.3) +
  geom_point(aes(cc, color=as.factor(cc))) +
  ggtitle("Clustering coeff. (real vs theoretical)") + 
  xlab('probability') +
  ylab('clustering coeff')

p1
dev.off()

pValues <- unique(data$p)

results <- data.frame(prob = numeric(), mVar = numeric())
for(p in pValues){
  values <- data[which(data$p == p),]
  meanVar <- abs(sum(values$ecc - values$cc))/nrow(values)
  results <- rbind(results, data.frame(p, meanVar))
}


pdf('./results/plot50_100_t.pdf')
p2 <- ggplot(results, aes(x=p, y=meanVar)) +
  geom_line(alpha=.3, color = I('red')) +
  ggtitle("Clustering coeff.") + 
  xlab('probability') +
  ylab('clustering coeff (mean variance)')

p2
dev.off()
