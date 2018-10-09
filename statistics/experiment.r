library(lattice)
library(PMCMR)

precision <- head(read.csv('../output/precision.csv', sep=';', header=T), 10)[,2:9]
recall <- head(read.csv('../output/recall.csv', sep=';', header=T), 10)[,2:9]
fscore <- head(read.csv('../output/fscore.csv', sep=';', header=T), 10)[,2:9]

print('Precision')
count <- 0
for (i in names(fscore)) {
    count <- count + 1
    png(paste('precision/', count, '-precision-', i, '.png', sep=''))
    par(cex.main=2, cex.lab=1.5, cex.axis=1.5)
    plot(density(precision[,i]), main=i, xlim=c(-0, 1))
    dev.off()
    print(i)
}
print('Precision General')
prec <- data.frame()
prec[1:10, 'C5.0 Target'] <- precision[1:10, 1]
prec[11:20, 'C5.0 Target'] <- precision[1:10, 2]
prec[21:30, 'C5.0 Target'] <- precision[1:10, 3]
prec[31:40, 'C5.0 Target'] <- precision[1:10, 4]
prec[1:10, 'C5.0 Crosscheck'] <- precision[1:10, 5]
prec[11:20, 'C5.0 Crosscheck'] <- precision[1:10, 6]
prec[21:30, 'C5.0 Crosscheck'] <- precision[1:10, 7]
prec[31:40, 'C5.0 Crosscheck'] <- precision[1:10, 8]
s_t <- shapiro.test(prec[,'C5.0 Target'])
print(s_t)
s_c <- shapiro.test(prec[,'C5.0 Crosscheck'])
print(s_c)

#print('Precision mann whitney')
#m <- wilcox.test(prec[,'C5.0 Target'], prec[, 'C5.0 Crosscheck'], paired=T)
print('T-Test')
m <- t.test(prec[,'C5.0 Target'], prec[, 'C5.0 Crosscheck'], paired=T)
print(m)

print('Recall')
count <- 0
for (i in names(fscore)) {
    count <- count + 1
    png(paste('recall/', count, '-recall-', i, '.png', sep=''))
    par(cex.main=2, cex.lab=1.5, cex.axis=1.5)
    plot(density(recall[,i]), main=i, xlim=c(0,1))
    dev.off()
    print(i)
}
print('Recall General')
rec <- data.frame()
rec[1:10, 'C5.0 Target'] <- recall[1:10, 1]
rec[11:20, 'C5.0 Target'] <- recall[1:10, 2]
rec[21:30, 'C5.0 Target'] <- recall[1:10, 3]
rec[31:40, 'C5.0 Target'] <- recall[1:10, 4]
rec[1:10, 'C5.0 Crosscheck'] <- recall[1:10, 5]
rec[11:20, 'C5.0 Crosscheck'] <- recall[1:10, 6]
rec[21:30, 'C5.0 Crosscheck'] <- recall[1:10, 7]
rec[31:40, 'C5.0 Crosscheck'] <- recall[1:10, 8]
s_t <- shapiro.test(rec[,'C5.0 Target'])
print(s_t)
s_c <- shapiro.test(rec[,'C5.0 Crosscheck'])
print(s_c)

print('Recall mann whitney')
m <- wilcox.test(rec[,'C5.0 Target'], rec[, 'C5.0 Crosscheck'], paired=T)
print(m)

print('F-measure')
count <- 0
for (i in names(fscore)) {
    count <- count + 1
    png(paste('fscore/', count, '-fscore-', i, '.png', sep=''))
    par(cex.main=2, cex.lab=1.5, cex.axis=1.5)
    plot(density(fscore[,i]), main=i, xlim=c(0,1))
    dev.off()
    print(i)
}
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

print('F-Measure mann whitney')
m <- wilcox.test(fmeasure[,'C5.0 Target'], fmeasure[, 'C5.0 Crosscheck'], paired=T)
print(m)
