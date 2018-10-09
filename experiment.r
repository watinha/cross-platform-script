library(randomForest)
library(C50)
library(e1071)
library(RWeka)
library(caret)
source('lib/learning-curve.r')
source('lib/cross-validation.r')
source('lib/feature-scale.r')

dataset_learning <- read.arff('data/all.filtered.analysed.arff')
dataset_learning[,'crosscheck.base.a'] <- dataset_learning[,'baseHeight'] * dataset_learning[,'baseWidth']
dataset_learning[,'crosscheck.target.a'] <- dataset_learning[,'targetHeight'] * dataset_learning[,'targetWidth']
dataset_learning[,'crosscheck.SDR'] <- abs(dataset_learning[,'crosscheck.base.a'] - dataset_learning[,'crosscheck.target.a']) /
                              pmax(pmin(dataset_learning[,'crosscheck.base.a'],dataset_learning[,'crosscheck.target.a']), 1)
dataset_learning[,'crosscheck.disp'] <- sqrt((dataset_learning[,'baseX'] - dataset_learning[,'targetX'])^2 +
                                             (dataset_learning[,'baseY'] - dataset_learning[,'targetY'])^2)
dataset_learning[,'crosscheck.area'] <- pmin(dataset_learning[,'crosscheck.base.a'], dataset_learning[,'crosscheck.target.a'])
dataset_learning[,'crosscheck.LDTD'] <- dataset_learning[,'baseViewportWidth'] / dataset_learning[,'baseViewportWidth']

features_crosscheck <- c(
    'crosscheck.SDR', 'crosscheck.disp', 'crosscheck.LDTD',
    'crosscheck.area', 'chiSquared'
)

dataset_learning[,'baseLeft'] <- dataset_learning[,'baseX']
dataset_learning[,'targetLeft'] <- dataset_learning[,'targetX']
dataset_learning[,'baseRight'] <- dataset_learning[,'baseViewportWidth'] - (dataset_learning[,'baseX'] + dataset_learning[,'baseWidth'])
dataset_learning[,'targetRight'] <- dataset_learning[,'targetViewportWidth'] - (dataset_learning[,'targetX'] + dataset_learning[,'targetWidth'])
dataset_learning[,'baseParentLeft'] <- dataset_learning[,'baseParentX']
dataset_learning[,'targetParentLeft'] <- dataset_learning[,'targetParentX']
dataset_learning[,'baseParentRight'] <- dataset_learning[,'baseViewportWidth'] - (dataset_learning[,'baseParentX'] + dataset_learning[,'baseWidth'])
dataset_learning[,'targetParentRight'] <- dataset_learning[,'targetViewportWidth'] - (dataset_learning[,'targetParentX'] + dataset_learning[,'targetWidth'])
dataset_learning[,'diff.viewport'] <- dataset_learning[,'targetViewportWidth'] - dataset_learning[,'baseViewportWidth']
dataset_learning[,'size'] <- dataset_learning[,'baseHeight'] * dataset_learning[,'baseWidth']

dataset_learning[,'diff.left.relation'] <- abs(dataset_learning[,'baseLeft']/dataset_learning[,'baseViewportWidth']) -
                                  abs(dataset_learning[,'targetLeft']/dataset_learning[,'targetViewportWidth'])
dataset_learning[,'diff.right.relation'] <- abs(dataset_learning[,'baseRight']/dataset_learning[,'baseViewportWidth']) -
                                  abs(dataset_learning[,'targetRight']/dataset_learning[,'targetViewportWidth'])
dataset_learning[,'diff.parent.left.relation'] <- abs(dataset_learning[,'baseParentLeft']/dataset_learning[,'baseViewportWidth']) -
                                         abs(dataset_learning[,'targetParentLeft']/dataset_learning[,'targetViewportWidth'])
dataset_learning[,'diff.parent.right.relation'] <- abs(dataset_learning[,'baseParentRight']/dataset_learning[,'baseViewportWidth']) -
                                          abs(dataset_learning[,'targetParentRight']/dataset_learning[,'targetViewportWidth'])
dataset_learning[,'diff.left.viewport'] <- abs(dataset_learning[,'baseLeft'] - dataset_learning[,'targetLeft']) / dataset_learning[,'diff.viewport']
dataset_learning[,'diff.right.viewport'] <- abs(dataset_learning[,'baseRight'] - dataset_learning[,'targetRight']) / dataset_learning[,'diff.viewport']
dataset_learning[,'alignment'] <- dataset_learning[,'diff.left.viewport'] - dataset_learning[,'diff.right.viewport']
dataset_learning[,'diff.parent.left.viewport'] <- abs(dataset_learning[,'baseParentLeft'] - dataset_learning[,'targetParentLeft']) / dataset_learning[,'diff.viewport']
dataset_learning[,'diff.parent.right.viewport'] <- abs(dataset_learning[,'baseParentRight'] - dataset_learning[,'targetParentRight']) / dataset_learning[,'diff.viewport']
dataset_learning[,'parent.alignment'] <- dataset_learning[,'diff.left.viewport'] - dataset_learning[,'diff.right.viewport']
dataset_learning[,'diff.parent.y'] <- abs(dataset_learning[,'baseParentY'] - dataset_learning[,'targetParentY']) / dataset_learning[,'diff.viewport']
dataset_learning[,'diff.height.height'] <- abs(dataset_learning[,'baseHeight'] - dataset_learning[,'targetHeight']) / pmax(dataset_learning[,'baseHeight'],1)
dataset_learning[,'diff.width.viewport'] <- abs(dataset_learning[,'baseWidth'] - dataset_learning[,'targetWidth']) / dataset_learning[,'diff.viewport']
dataset_learning[,'imageDiff.size'] <- dataset_learning[,'imageDiff'] / pmax(dataset_learning[,'size'], 1)
dataset_learning[,'chiSquared.size'] <- dataset_learning[,'chiSquared'] / pmax(dataset_learning[,'size'], 1)
dataset_learning[,'size'] <- pmax(dataset_learning[,'baseHeight'] * dataset_learning[,'baseWidth'], dataset_learning[,'targetHeight'] * dataset_learning[,'targetWidth'])
dataset_learning[,'base.out.viewport.right'] <- dataset_learning[,'baseLeft'] - dataset_learning[,'baseViewportWidth']
dataset_learning[,'base.out.viewport.left'] <- dataset_learning[,'baseRight'] - dataset_learning[,'baseViewportWidth']
dataset_learning[,'target.out.viewport.right'] <- dataset_learning[,'targetLeft'] - dataset_learning[,'targetViewportWidth']
dataset_learning[,'target.out.viewport.left'] <- dataset_learning[,'targetRight'] - dataset_learning[,'targetViewportWidth']
dataset_learning[,'diff.out.viewport.left'] <- abs(
        dataset_learning[,'base.out.viewport.left'] - dataset_learning[,'target.out.viewport.left'])
dataset_learning[,'diff.out.viewport.right'] <- abs(
        dataset_learning[,'base.out.viewport.right'] - dataset_learning[,'target.out.viewport.right'])

features_target <- c(
#    'baseHeight', 'baseWidth', 'targetHeight', 'targetWidth',
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
    'imageDiff.size',
    'chiSquared.size',
    'phash',
    'childsNumber', 'textLength',
    'diff.out.viewport.left', 'diff.out.viewport.right'
)

c5_boosted20 <- function (X, y, X_cv, y_cv) {
    model <- C5.0(X, y, trials=20, control = C5.0Control(winnow=TRUE, noGlobalPruning=FALSE));
    cv_predictions <- predict.C5.0(model, X_cv)
    return (table(y_cv, cv_predictions))
}
c5_boosted10 <- function (X, y, X_cv, y_cv) {
    model <- C5.0(X, y, trials=10, control = C5.0Control(winnow=TRUE, noGlobalPruning=FALSE));
    cv_predictions <- predict.C5.0(model, X_cv)
    return (table(y_cv, cv_predictions))
}
c5_boosted5 <- function (X, y, X_cv, y_cv) {
    model <- C5.0(X, y, trials=5, control = C5.0Control(winnow=TRUE, noGlobalPruning=FALSE));
    cv_predictions <- predict.C5.0(model, X_cv)
    return (table(y_cv, cv_predictions))
}
c5 <- function (X, y, X_cv, y_cv) {
    model <- C5.0(X, y, trials=1, control = C5.0Control(winnow=TRUE, noGlobalPruning=FALSE));
    cv_predictions <- predict.C5.0(model, X_cv)
    return (table(y_cv, cv_predictions))
}
svm_std <- function (X, y, X_cv, y_cv) {
    model <- svm(X, y, scale=T, probability=TRUE)
    cv_predictions <- predict(model, X_cv)
    return (table(y_cv, cv_predictions))
}
svm_std_cutoff <- function (X, y, X_cv, y_cv) {
    model <- svm(X, y, scale=T, probability=TRUE)
    cv_predictions <- ifelse(attr(predict(model, X_cv, probability=T), 'probabilities')[,1] >= 0.2, 1, 0)
    return (table(y_cv, cv_predictions))
}
svm_linear <- function (X, y, X_cv, y_cv) {
    model <- svm(X, y, type="C-classification", kernel="linear", cost=1000, probability=TRUE)
    cv_predictions <- ifelse(attr(predict(model, X_cv, threshold=0, probability=TRUE), "probabilities")[,1] >= 0.3198411, 1, 0)
    return (table(y_cv, cv_predictions))
}
svm_polynomial <- function (X, y, X_cv, y_cv) {
    svm_model <- svm(X, y, type="C-classification", kernel="polynomial", degree=2, gamma=0.1, coef0=1000, cost=100, probability=TRUE)
    cv_predictions <- ifelse(attr(predict(svm_model, X_cv, threshold=0, probability=TRUE), "probabilities")[,1] >= 0.1281602, 1, 0)
    return (table(y_cv, cv_predictions))
}
random_forest <- function (X, y, X_cv, y_cv) {
    model <- randomForest(X, y)
    cv_predictions <- predict(model, X_cv)
    return (table(y_cv, cv_predictions))
}

#r1 <- learningCurve(dataset_learning, dataset_test, features9, decision_tree)

#X <- dataset_learning[,features7[1:(length(features7) - 1)]]
#y <- dataset_learning[,'Result']
#model <- C5.0(X, y, control = C5.0Control(winnow=TRUE, noGlobalPruning=FALSE));
#summary(model)

set.seed(42)

urls <- unique(dataset_learning[,'URL'])
r <- matrix(nrow=103, ncol=10)
p <- matrix(nrow=103, ncol=10)
rec <- matrix(nrow=103, ncol=10)
iterations <- 0

for (i in 1:1) {
    iterations <- 1
    f_urls <- sample(urls)
    folds <- createFolds(dataset_learning[, 'Result'])
    folds$Fold01 <- as.numeric(rownames(subset(dataset_learning, URL %in% f_urls[1:4])))
    folds$Fold02 <- as.numeric(rownames(subset(dataset_learning, URL %in% f_urls[5:8])))
    folds$Fold03 <- as.numeric(rownames(subset(dataset_learning, URL %in% f_urls[9:12])))
    folds$Fold04 <- as.numeric(rownames(subset(dataset_learning, URL %in% f_urls[13:16])))
    folds$Fold05 <- as.numeric(rownames(subset(dataset_learning, URL %in% f_urls[17:20])))
    folds$Fold06 <- as.numeric(rownames(subset(dataset_learning, URL %in% f_urls[21:24])))
    folds$Fold07 <- as.numeric(rownames(subset(dataset_learning, URL %in% f_urls[25:28])))
    folds$Fold08 <- as.numeric(rownames(subset(dataset_learning, URL %in% f_urls[29:34])))
    folds$Fold09 <- as.numeric(rownames(subset(dataset_learning, URL %in% f_urls[35:40])))
    folds$Fold10 <- as.numeric(rownames(subset(dataset_learning, URL %in% f_urls[41:45])))

    print('evaluating... ')
    print(i)
    print(f_urls)
    c50_1 <- crossValidation(dataset_learning, dataset_learning, folds, features_target, svm_std, 'output/1.csv');
    c50_2 <- crossValidation(dataset_learning, dataset_learning, folds, features_target, random_forest, 'output/2.csv');
    c50_3 <- crossValidation(dataset_learning, dataset_learning, folds, features_target, c5_boosted20, 'output/3.csv');
    c50_4 <- crossValidation(dataset_learning, dataset_learning, folds, features_target, c5, 'output/4.csv');
    c50_5 <- crossValidation(dataset_learning, dataset_learning, folds, features_crosscheck, svm_std, 'output/5.csv');
    c50_6 <- crossValidation(dataset_learning, dataset_learning, folds, features_crosscheck, random_forest, 'output/6.csv');
    c50_7 <- crossValidation(dataset_learning, dataset_learning, folds, features_crosscheck, c5_boosted20, 'output/7.csv');
    c50_8 <- crossValidation(dataset_learning, dataset_learning, folds, features_crosscheck, c5, 'output/8.csv');
    c50_9 <- crossValidation(dataset_learning, dataset_learning, folds, features_target, svm_std_cutoff, 'output/9.csv');

    r[((i - 1)*10 + 2):(10*(i) + 1), 1] <- c('Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5',
                                         'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9', 'Fold 10')
    r[((i - 1)*10 + 2):(10*(i) + 1), 2] <- c50_1[3:12,8]
    r[((i - 1)*10 + 2):(10*(i) + 1), 3] <- c50_2[3:12,8]
    r[((i - 1)*10 + 2):(10*(i) + 1), 4] <- c50_3[3:12,8]
    r[((i - 1)*10 + 2):(10*(i) + 1), 5] <- c50_4[3:12,8]
    r[((i - 1)*10 + 2):(10*(i) + 1), 6] <- c50_5[3:12,8]
    r[((i - 1)*10 + 2):(10*(i) + 1), 7] <- c50_6[3:12,8]
    r[((i - 1)*10 + 2):(10*(i) + 1), 8] <- c50_7[3:12,8]
    r[((i - 1)*10 + 2):(10*(i) + 1), 9] <- c50_8[3:12,8]
    r[((i - 1)*10 + 2):(10*(i) + 1), 10] <- c50_9[3:12,8]

    p[((i - 1)*10 + 2):(10*(i) + 1), 1] <- c('Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5',
                                         'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9', 'Fold 10')
    p[((i - 1)*10 + 2):(10*(i) + 1), 2] <- c50_1[3:12,6]
    p[((i - 1)*10 + 2):(10*(i) + 1), 3] <- c50_2[3:12,6]
    p[((i - 1)*10 + 2):(10*(i) + 1), 4] <- c50_3[3:12,6]
    p[((i - 1)*10 + 2):(10*(i) + 1), 5] <- c50_4[3:12,6]
    p[((i - 1)*10 + 2):(10*(i) + 1), 6] <- c50_5[3:12,6]
    p[((i - 1)*10 + 2):(10*(i) + 1), 7] <- c50_6[3:12,6]
    p[((i - 1)*10 + 2):(10*(i) + 1), 8] <- c50_7[3:12,6]
    p[((i - 1)*10 + 2):(10*(i) + 1), 9] <- c50_8[3:12,6]
    p[((i - 1)*10 + 2):(10*(i) + 1), 10] <- c50_9[3:12,6]

    rec[((i - 1)*10 + 2):(10*(i) + 1), 1] <- c('Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5',
                                           'Fold 6', 'Fold 7', 'Fold 8', 'Fold 9', 'Fold 10')
    rec[((i - 1)*10 + 2):(10*(i) + 1), 2] <- c50_1[3:12,7]
    rec[((i - 1)*10 + 2):(10*(i) + 1), 3] <- c50_2[3:12,7]
    rec[((i - 1)*10 + 2):(10*(i) + 1), 4] <- c50_3[3:12,7]
    rec[((i - 1)*10 + 2):(10*(i) + 1), 5] <- c50_4[3:12,7]
    rec[((i - 1)*10 + 2):(10*(i) + 1), 6] <- c50_5[3:12,7]
    rec[((i - 1)*10 + 2):(10*(i) + 1), 7] <- c50_6[3:12,7]
    rec[((i - 1)*10 + 2):(10*(i) + 1), 8] <- c50_7[3:12,7]
    rec[((i - 1)*10 + 2):(10*(i) + 1), 9] <- c50_8[3:12,7]
    rec[((i - 1)*10 + 2):(10*(i) + 1), 10] <- c50_9[3:12,7]

}
for (i in 2:10) {
    r[iterations*10 + 2, i] <- mean(as.numeric(r[2:(iterations*10+1), i]))
    p[iterations*10 + 2, i] <- mean(as.numeric(p[2:(iterations*10+1), i]))
    rec[iterations*10 + 2, i] <- mean(as.numeric(rec[2:(iterations*10+1), i]))
    r[iterations*10 + 3, i] <- sd(as.numeric(r[2:(iterations*10+1), i]))
    p[iterations*10 + 3, i] <- sd(as.numeric(p[2:(iterations*10+1), i]))
    rec[iterations*10 + 3, i] <- sd(as.numeric(rec[2:(iterations*10+1), i]))
}
r[(iterations*10+2), 1] <- 'Mean'
p[(iterations*10+2), 1] <- 'Mean'
rec[(iterations*10+2), 1] <- 'Mean'
p[(iterations*10+3), 1] <- 'SD'
r[(iterations*10+3), 1] <- 'SD'
rec[(iterations*10+3), 1] <- 'SD'

r[1, ] <- c('', 'SVM-target', 'rf-target', 'c5-boosted20-target', 'c5-target', 'SVM-crosscheck', 'rf-crosscheck', 'c5-boosted20-crosscheck', 'c5-crosscheck', 'SVM-target')
write.table(r, file='output/fscore.csv', quote=FALSE, sep=';', row.names=FALSE, col.names=FALSE)
p[1, ] <- c('', 'SVM-target', 'rf-target', 'c5-boosted20-target', 'c5-target', 'SVM-crosscheck', 'rf-crosscheck', 'c5-boosted20-crosscheck', 'c5-crosscheck', 'SVM-target')
write.table(p, file='output/precision.csv', quote=FALSE, sep=';', row.names=FALSE, col.names=FALSE)
rec[1, ] <- c('', 'SVM-target', 'rf-target', 'c5-boosted20-target', 'c5-target', 'SVM-crosscheck', 'rf-crosscheck', 'c5-boosted20-crosscheck', 'c5-crosscheck', 'SVM-target')
write.table(rec, file='output/recall.csv', quote=FALSE, sep=';', row.names=FALSE, col.names=FALSE)
