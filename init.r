library(caret)
library(RWeka)
library(C50)
library(e1071)
library(ROCR)
source('lib/learning-curve.r');

dataset <- read.arff('data/all.filtered.analysed.arff')
dataset[,'crosscheck.base.a'] <- dataset[,'baseHeight'] * dataset[,'baseWidth']
dataset[,'crosscheck.target.a'] <- dataset[,'targetHeight'] * dataset[,'targetWidth']
dataset[,'crosscheck.SDR'] <- abs(dataset[,'crosscheck.base.a'] - dataset[,'crosscheck.target.a']) /
                              pmax(pmin(dataset[,'crosscheck.base.a'],dataset[,'crosscheck.target.a']), 1)
dataset[,'crosscheck.disp'] <- sqrt((dataset[,'baseX'] - dataset[,'targetX'])^2 +
                                             (dataset[,'baseY'] - dataset[,'targetY'])^2)
dataset[,'crosscheck.area'] <- pmin(dataset[,'crosscheck.base.a'], dataset[,'crosscheck.target.a'])
dataset[,'crosscheck.LDTD'] <- dataset[,'baseViewportWidth'] / dataset[,'baseViewportWidth']

dataset[,'baseLeft'] <- dataset[,'baseX']
dataset[,'targetLeft'] <- dataset[,'targetX']
dataset[,'baseRight'] <- dataset[,'baseViewportWidth'] - (dataset[,'baseX'] + dataset[,'baseWidth'])
dataset[,'targetRight'] <- dataset[,'targetViewportWidth'] - (dataset[,'targetX'] + dataset[,'targetWidth'])
dataset[,'baseParentLeft'] <- dataset[,'baseParentX']
dataset[,'targetParentLeft'] <- dataset[,'targetParentX']
dataset[,'baseParentRight'] <- dataset[,'baseViewportWidth'] - (dataset[,'baseParentX'] + dataset[,'baseWidth'])
dataset[,'targetParentRight'] <- dataset[,'targetViewportWidth'] - (dataset[,'targetParentX'] + dataset[,'targetWidth'])
dataset[,'diff.viewport'] <- dataset[,'targetViewportWidth'] - dataset[,'baseViewportWidth']
dataset[,'size'] <- dataset[,'baseHeight'] * dataset[,'baseWidth']
dataset[,'base.out.viewport.right'] <- dataset[,'baseLeft'] - dataset[,'baseViewportWidth']
dataset[,'base.out.viewport.left'] <- dataset[,'baseRight'] - dataset[,'baseViewportWidth']
dataset[,'target.out.viewport.right'] <- dataset[,'targetLeft'] - dataset[,'targetViewportWidth']
dataset[,'target.out.viewport.left'] <- dataset[,'targetRight'] - dataset[,'targetViewportWidth']
dataset[,'diff.out.viewport.left'] <- abs(dataset[,'base.out.viewport.left'] - dataset[,'target.out.viewport.left'])
dataset[,'diff.out.viewport.right'] <- abs(dataset[,'base.out.viewport.right'] - dataset[,'target.out.viewport.right'])

dataset[,'diff.left.relation'] <- abs(dataset[,'baseLeft']/dataset[,'baseViewportWidth']) -
                                  abs(dataset[,'targetLeft']/dataset[,'targetViewportWidth'])
dataset[,'diff.right.relation'] <- abs(dataset[,'baseRight']/dataset[,'baseViewportWidth']) -
                                  abs(dataset[,'targetRight']/dataset[,'targetViewportWidth'])
dataset[,'diff.parent.left.relation'] <- abs(dataset[,'baseParentLeft']/dataset[,'baseViewportWidth']) -
                                         abs(dataset[,'targetParentLeft']/dataset[,'targetViewportWidth'])
dataset[,'diff.parent.right.relation'] <- abs(dataset[,'baseParentRight']/dataset[,'baseViewportWidth']) -
                                          abs(dataset[,'targetParentRight']/dataset[,'targetViewportWidth'])
dataset[,'diff.left.viewport'] <- abs(dataset[,'baseLeft'] - dataset[,'targetLeft']) / dataset[,'diff.viewport']
dataset[,'diff.right.viewport'] <- abs(dataset[,'baseRight'] - dataset[,'targetRight']) / dataset[,'diff.viewport']
dataset[,'alignment'] <- dataset[,'diff.left.viewport'] - dataset[,'diff.right.viewport']
dataset[,'diff.parent.left.viewport'] <- abs(dataset[,'baseParentLeft'] - dataset[,'targetParentLeft']) / dataset[,'diff.viewport']
dataset[,'diff.parent.right.viewport'] <- abs(dataset[,'baseParentRight'] - dataset[,'targetParentRight']) / dataset[,'diff.viewport']
dataset[,'parent.alignment'] <- dataset[,'diff.left.viewport'] - dataset[,'diff.right.viewport']
dataset[,'diff.parent.y'] <- abs(dataset[,'baseParentY'] - dataset[,'targetParentY']) / dataset[,'diff.viewport']
dataset[,'diff.parent.y.height'] <- abs(dataset[,'baseParentY'] - dataset[,'targetParentY']) / pmax(dataset[,'baseHeight'],1)
dataset[,'diff.height.height'] <- abs(dataset[,'baseHeight'] - dataset[,'targetHeight']) / pmax(dataset[,'baseHeight'],1)
dataset[,'diff.width.viewport'] <- abs(dataset[,'baseWidth'] - dataset[,'targetWidth']) / dataset[,'diff.viewport']
dataset[,'imageDiff.size'] <- dataset[,'imageDiff'] / pmax(dataset[,'size'], 1)
dataset[,'chiSquared.size'] <- dataset[,'chiSquared'] / pmax(dataset[,'size'], 1)
dataset[,'size'] <- pmax(dataset[,'baseHeight'] * dataset[,'baseWidth'], dataset[,'targetHeight'] * dataset[,'targetWidth'])
dataset[,'imageDiff.size.viewport'] <- dataset[,'imageDiff.size'] / dataset[,'diff.viewport']
dataset[,'chiSquared.size.viewport'] <- dataset[,'chiSquared.size'] / dataset[,'diff.viewport']
dataset[,'pHashDistance.viewport'] <- dataset[,'phash'] / dataset[,'diff.viewport']

features_target <- c(
    'size',
    'diff.height.height',
#    'diff.left.relation', 'diff.right.relation',
#    'diff.parent.left.relation', 'diff.parent.right.relation',
    'diff.width.viewport',
    'diff.left.viewport',
    'diff.right.viewport',
#    'alignment',
#    'diff.parent.left.viewport',
#    'diff.parent.right.viewport',
#    'parent.alignment',
    'diff.parent.y',
#    'imageDiff.size.viewport',
#    'chiSquared.size.viewport',
#    'pHashDistance.viewport'
    'imageDiff.size',
    'chiSquared.size',
    'phash',
    'childsNumber',
    'textLength',
    'diff.out.viewport.left', 'diff.out.viewport.right'
)
features_target_svm <- c(
#    'size',
#    'baseX', 'targetX', 'baseY', 'targetY',
#    'baseHeight', 'targetHeight', 'baseWidth', 'targetWidth',
#    'baseParentX', 'targetParentX', 'baseParentY', 'targetParentY',
#    'baseLeft', 'baseParentLeft', 'targetLeft', 'targetParentLeft',
#    'baseRight', 'baseParentRight', 'targetRight', 'targetParentRight',
#    'baseViewportWidth', 'targetViewportWidth',
#    'imageDiff', 'chiSquared', 'phash',
#    'childsNumber', 'textLength'
    'size',
    'diff.height.height',
#    'diff.left.relation', 'diff.right.relation',
#    'diff.parent.left.relation', 'diff.parent.right.relation',
    'diff.width.viewport',
    'diff.left.viewport',
    'diff.right.viewport',
#    'alignment',
    'diff.parent.left.viewport',
    'diff.parent.right.viewport',
#    'parent.alignment',
    'diff.parent.y',
    'imageDiff.size',
    'chiSquared.size',
    'phash',
    'childsNumber', 'textLength',
    'diff.out.viewport.left', 'diff.out.viewport.right'
)
features_crosscheck <- c(
    'crosscheck.SDR', 'crosscheck.disp', 'crosscheck.LDTD',
    'crosscheck.area', 'chiSquared'
)
#set.seed(42)
urls <- unique(dataset[,'URL'])
f_urls <- sample(urls)
folds <- createFolds(dataset[, 'Result'], k = 10)
folds$Fold01 <- as.numeric(rownames(subset(dataset, URL %in% f_urls[1:3])))
folds$Fold02 <- as.numeric(rownames(subset(dataset, URL %in% f_urls[4:6])))
folds$Fold03 <- as.numeric(rownames(subset(dataset, URL %in% f_urls[7:9])))
folds$Fold04 <- as.numeric(rownames(subset(dataset, URL %in% f_urls[12:15])))
folds$Fold05 <- as.numeric(rownames(subset(dataset, URL %in% f_urls[16:18])))
folds$Fold06 <- as.numeric(rownames(subset(dataset, URL %in% f_urls[19:21])))
folds$Fold07 <- as.numeric(rownames(subset(dataset, URL %in% f_urls[22:24])))
folds$Fold08 <- as.numeric(rownames(subset(dataset, URL %in% f_urls[25:28])))
folds$Fold09 <- as.numeric(rownames(subset(dataset, URL %in% f_urls[29:32])))
folds$Fold10 <- as.numeric(rownames(subset(dataset, URL %in% f_urls[33:45])))
dataset_train <- dataset[-folds$Fold10,]
dataset_cv <- dataset[folds$Fold10,]

#X_train_target <- dataset[,features_target]
X_train_target <- dataset_train[,features_target]
X_cv_target <- dataset_cv[,features_target]
X_train_base <- dataset_train[,features_crosscheck]
X_cv_base <- dataset_cv[,features_crosscheck]
X_train_svm <- dataset_train[,features_target_svm]
X_cv_svm <- dataset_cv[,features_target_svm]

#y_train <- dataset[,'Result']
y_train <- dataset_train[,'Result']
y_cv <- dataset_cv[,'Result']

cost <- matrix(c(1,1,1,1), nrow=2, ncol=2, byrow=T)
dimnames(cost) <- list(c(0, 1), c(0, 1))

print(features_target)
m_target <- C5.0(X_train_target, y_train, trials=1, control=C5.0Control(winnow=T, CF=0.25, minCases=2))
h <- predict(m_target, X_cv_target)
print(table(y_train, predict(m_target, X_train_target)))
confusion_table <- table(y_cv, h)
print(confusion_table)
print(features_crosscheck)
m_base <- C5.0(X_train_base, y_train, trials=1, control=C5.0Control(winnow=T, CF=0.25, minCases=2))
print(table(y_train, predict(m_base, X_train_base)))
print(table(y_cv, predict(m_base, X_cv_base)))
#print('---SVM---')
#print(features_target_svm)
#m_svm_target <- svm(X_train_svm, y_train, scale=T, probability=T)
#print(table(y_train, predict(m_svm_target, X_train_svm)))
#print(table(y_cv, predict(m_svm_target, X_cv_svm)))
#
#y_pred <- attr(predict(m_svm_target, X_cv_svm, probability=TRUE), 'probabilities')[,'1']
#pred <- prediction(y_pred, y_cv)
#perf_tpr <- performance(pred, 'tpr', 'fpr')
#perf_f <- performance(pred, 'f')
#png('roc-tpr-target.png')
#plot(perf_tpr)
#dev.off()
#png('roc-f-target.png')
#plot(perf_f)
#dev.off()

#m_svm_base <- svm(X_base, y, cross=10, probability=TRUE)
#y_pred <- attr(predict(m_svm_base, X_base, probability=TRUE), 'probabilities')[,'1']
#pred <- prediction(y_pred, y)
#perf_tpr <- performance(pred, 'tpr', 'fpr')
#perf_f <- performance(pred, 'f')
#png('roc-tpr-base.png')
#plot(perf_tpr)
#dev.off()
#png('roc-f-base.png')
#plot(perf_f)
#dev.off()

#c5 <- function (X, y, X_cv, y_cv) {
#    model <- C5.0(X, y, trials=1, control = C5.0Control(winnow=TRUE, noGlobalPruning=FALSE));
#    cv_predictions <- predict.C5.0(model, X_cv)
#    return (table(y_cv, cv_predictions))
#}
#learningCurve(dataset_train, dataset_cv, features_target, c5);
