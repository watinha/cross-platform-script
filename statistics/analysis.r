library(lattice)
library(PMCMR)

#methods <- c(3, 4, 5, 7, 8, 9)

#precision <- head(read.csv('../output/precision.csv', sep=';', header=T), 10)[,2:9]
#recall <- head(read.csv('../output/recall.csv', sep=';', header=T), 10)[,2:9]
fscore <- head(read.csv('../output/fscore.csv', sep=';', header=T), 10)[,2:9]

print('F-measure')
print('F-measure - ShapiroWilk')
count <- 0
for (i in names(fscore)) {
    count <- count + 1
    png(paste(count, '-fscore-', i, '.png', sep=''))
    plot(density(fscore[,i]), main=i)
    dev.off()
    print(i)
    r <- shapiro.test(fscore[,i])
    print(r)
}

print('F-Measure friedman test')
r <- friedman.test(as.matrix(fscore))
print(r)
summary(r)
print('F-Measure posthoc test')
post <- posthoc.friedman.nemenyi.test(as.matrix(fscore))
print(post)

print('F-Measure t-tests')
t <- t.test(fscore[,1], fscore[,5], paired=T)
print(t)
t <- t.test(fscore[,2], fscore[,6], paired=T)
print(t)
t <- t.test(fscore[,3], fscore[,7], paired=T)
print(t)
t <- t.test(fscore[,4], fscore[,8], paired=T)
print(t)
print('F-measure General t.test')
fmeasure <- data.frame()
fmeasure[1:10, 'C5.0 Target'] <- fscore[1:10, 1]
fmeasure[11:20, 'C5.0 Target'] <- fscore[1:10, 2]
fmeasure[21:30, 'C5.0 Target'] <- fscore[1:10, 3]
fmeasure[31:40, 'C5.0 Target'] <- fscore[1:10, 4]
fmeasure[1:10, 'C5.0 Crosscheck'] <- fscore[1:10, 5]
fmeasure[11:20, 'C5.0 Crosscheck'] <- fscore[1:10, 6]
fmeasure[21:30, 'C5.0 Crosscheck'] <- fscore[1:10, 7]
fmeasure[31:40, 'C5.0 Crosscheck'] <- fscore[1:10, 8]
s_t <- shapiro.test(fmeasure[,'C5.0 Target'])
print(s_t)
s_c <- shapiro.test(fmeasure[,'C5.0 Crosscheck'])
print(s_c)
t <- t.test(fmeasure[,'C5.0 Target'], fmeasure[, 'C5.0 Crosscheck'])
print(t)


print('F-Measure mann whitney')
m <- wilcox.test(fmeasure[,'C5.0 Target'], fmeasure[, 'C5.0 Crosscheck'], paired=T)
print(m)

