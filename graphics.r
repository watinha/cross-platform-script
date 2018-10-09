library(e1071)
library(caret)
library(C50)
library(ROCR)
library(RWeka)
source('lib/learning-curve.r')

dataset <- read.arff('data/all.arff')
dataset[,'diff.x'] <- abs(dataset[,'baseX'] - dataset[,'targetX'])
dataset[,'diff.y'] <- abs(dataset[,'baseY'] - dataset[,'targetY'])
dataset[,'diff.height'] <- abs(dataset[,'baseHeight'] - dataset[,'targetHeight'])
dataset[,'diff.width'] <- abs(dataset[,'baseWidth'] - dataset[,'targetWidth'])
dataset[,'diff.parent.x'] <- abs(dataset[,'baseParentX'] - dataset[,'targetParentX'])
dataset[,'diff.parent.y'] <- abs(dataset[,'baseParentY'] - dataset[,'targetParentY'])
dataset[,'diff.x.dpi'] <- abs(dataset[,'baseX']/dataset[,'baseDPI'] -
                                       dataset[,'targetX']/dataset[,'targetDPI'])
dataset[,'diff.y.dpi'] <- abs(dataset[,'baseY']/dataset[,'baseDPI'] -
                                       dataset[,'targetY']/dataset[,'targetDPI'])
dataset[,'diff.height.dpi'] <- abs(dataset[,'baseHeight']/dataset[,'baseDPI'] -
                                       dataset[,'targetHeight']/dataset[,'targetDPI'])
dataset[,'diff.width.dpi'] <- abs(dataset[,'baseWidth']/dataset[,'baseDPI'] -
                                       dataset[,'targetWidth']/dataset[,'targetDPI'])
dataset[,'diff.parent.x.dpi'] <- abs(dataset[,'baseParentX']/dataset[,'baseDPI'] -
                                       dataset[,'targetParentX']/dataset[,'targetDPI'])
dataset[,'diff.parent.y.dpi'] <- abs(dataset[,'baseParentY']/dataset[,'baseDPI'] -
                                       dataset[,'targetParentY']/dataset[,'targetDPI'])
dataset[,'diff.x.relation'] <- abs(dataset[,'baseX']/dataset[,'baseViewportWidth'] -
                                       dataset[,'targetX']/dataset[,'targetViewportWidth'])
dataset[,'diff.y.relation'] <- abs(dataset[,'baseY']/dataset[,'baseViewportWidth'] -
                                       dataset[,'targetY']/dataset[,'targetViewportWidth'])
dataset[,'diff.height.relation'] <- abs(dataset[,'baseHeight']/dataset[,'baseViewportWidth'] -
                                       dataset[,'targetHeight']/dataset[,'targetViewportWidth'])
dataset[,'diff.width.relation'] <- abs(dataset[,'baseWidth']/dataset[,'baseViewportWidth'] -
                                       dataset[,'targetWidth']/dataset[,'targetViewportWidth'])
dataset[,'diff.parent.x.relation'] <- abs(dataset[,'baseParentX']/dataset[,'baseViewportWidth'] -
                                       dataset[,'targetParentX']/dataset[,'targetViewportWidth'])
dataset[,'diff.parent.y.relation'] <- abs(dataset[,'baseParentY']/dataset[,'baseViewportWidth'] -
                                       dataset[,'targetParentY']/dataset[,'targetViewportWidth'])
dataset[,'diff.relation'] <- dataset[,'imageDiff'] / max(dataset[,'baseHeight'] *
                                                                           dataset[,'baseWidth'], 1)
dataset[,'chi.relation'] <- dataset[,'chiSquared'] / max(dataset[,'baseHeight'] *
                                                                           dataset[,'baseWidth'], 1)

features_3 <- c(
    'diff.x.relation', 'diff.y.relation',
    'diff.height.relation', 'diff.width.relation',
    'diff.parent.x.relation', 'diff.parent.y.relation',
    'diff.relation', 'chi.relation'
)

folds <- createFolds(dataset[,'Result'], k=2)
learning <- dataset[folds$Fold1,]
cv <- dataset[folds$Fold2,]
X <- learning[,features_3]
X_cv <- cv[,features_3]
y <- learning[,'Result']
y_cv <- cv[,'Result']

svm_model <- svm(X, y, type="C-classification", kernel="polynomial", degree=2, cost=100, gamma=0.1, coef0=1000, probability=TRUE)
y_pred <- attr(predict(svm_model, X_cv, probability=TRUE), 'probabilities')[,'1']
pred <- prediction(y_pred, y_cv)
perf_tpr <- performance(pred, 'tpr', 'fpr')
perf_f <- performance(pred, 'f')
png('roc-tpr-poly.png')
plot(perf_tpr)
dev.off()
png('roc-f-poly.png')
plot(perf_f)
dev.off()
