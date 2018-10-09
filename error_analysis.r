library(caret)
library(RWeka)
library(C50)
library(e1071)
library(ROCR)

dataset <- read.arff('data/all.filtered.analysed.arff')
dataset[,'crosscheck.base.a'] <- dataset[,'baseHeight'] * dataset[,'baseWidth']
dataset[,'crosscheck.target.a'] <- dataset[,'targetHeight'] * dataset[,'targetWidth']
dataset[,'crosscheck.SDR'] <- abs(dataset[,'crosscheck.base.a'] - dataset[,'crosscheck.target.a']) /
                              pmax(pmin(dataset[,'crosscheck.base.a'],dataset[,'crosscheck.target.a']), 1)
dataset[,'crosscheck.disp'] <- sqrt((dataset[,'baseX'] - dataset[,'targetX'])^2 +
                                             (dataset[,'baseY'] - dataset[,'targetY'])^2)
dataset[,'crosscheck.area'] <- pmin(dataset[,'crosscheck.base.a'], dataset[,'crosscheck.target.a'])
dataset[,'crosscheck.LDTD'] <- dataset[,'baseViewportWidth'] / dataset[,'baseViewportWidth']

features_all <- c(
    'baseX', 'targetX', 'baseY', 'targetY',
    'baseHeight', 'targetHeight', 'baseWidth', 'targetWidth',
    'baseParentX', 'targetParentX', 'baseParentY', 'targetParentY',
    'imageDiff', 'chiSquared',
    'baseViewportWidth', 'targetViewportWidth',
    'phash',
    'childsNumber', 'textLength'
)

dataset[,'baseLeft'] <- dataset[,'baseX']
dataset[,'targetLeft'] <- dataset[,'targetX']
dataset[,'baseRight'] <- dataset[,'baseViewportWidth'] - (dataset[,'baseX'] + dataset[,'baseWidth'])
dataset[,'targetRight'] <- dataset[,'targetViewportWidth'] - (dataset[,'targetX'] + dataset[,'targetWidth'])
#dataset[,'baseParentLeft'] <- dataset[,'baseParentX']
#dataset[,'targetParentLeft'] <- dataset[,'targetParentX']
#dataset[,'baseParentRight'] <- dataset[,'baseViewportWidth'] - (dataset[,'baseParentX'] + dataset[,'baseWidth'])
#dataset[,'targetParentRight'] <- dataset[,'targetViewportWidth'] - (dataset[,'targetParentX'] + dataset[,'targetWidth'])
dataset[,'diff.viewport'] <- dataset[,'targetViewportWidth'] - dataset[,'baseViewportWidth']
dataset[,'size'] <- dataset[,'baseHeight'] * dataset[,'baseWidth']

#dataset[,'diff.left.relation'] <- abs(dataset[,'baseLeft']/dataset[,'baseViewportWidth']) -
#                                  abs(dataset[,'targetLeft']/dataset[,'targetViewportWidth'])
#dataset[,'diff.right.relation'] <- abs(dataset[,'baseRight']/dataset[,'baseViewportWidth']) -
#                                  abs(dataset[,'targetRight']/dataset[,'targetViewportWidth'])
#dataset[,'diff.parent.left.relation'] <- abs(dataset[,'baseParentLeft']/dataset[,'baseViewportWidth']) -
#                                         abs(dataset[,'targetParentLeft']/dataset[,'targetViewportWidth'])
#dataset[,'diff.parent.right.relation'] <- abs(dataset[,'baseParentRight']/dataset[,'baseViewportWidth']) -
#                                          abs(dataset[,'targetParentRight']/dataset[,'targetViewportWidth'])
dataset[,'diff.left.viewport'] <- abs(dataset[,'baseLeft'] - dataset[,'targetLeft']) / dataset[,'diff.viewport']
dataset[,'diff.right.viewport'] <- abs(dataset[,'baseRight'] - dataset[,'targetRight']) / dataset[,'diff.viewport']
#dataset[,'alignment'] <- dataset[,'diff.left.viewport'] - dataset[,'diff.right.viewport']
#dataset[,'diff.parent.left.viewport'] <- abs(dataset[,'baseParentLeft'] - dataset[,'targetParentLeft']) / dataset[,'diff.viewport']
#dataset[,'diff.parent.right.viewport'] <- abs(dataset[,'baseParentRight'] - dataset[,'targetParentRight']) / dataset[,'diff.viewport']
#dataset[,'parent.alignment'] <- dataset[,'diff.left.viewport'] - dataset[,'diff.right.viewport']
dataset[,'diff.parent.y'] <- abs(dataset[,'baseParentY'] - dataset[,'targetParentY']) / dataset[,'diff.viewport']
#dataset[,'diff.parent.y.height'] <- abs(dataset[,'baseParentY'] - dataset[,'targetParentY']) / pmax(dataset[,'baseHeight'],1)
dataset[,'diff.height.height'] <- abs(dataset[,'baseHeight'] - dataset[,'targetHeight']) / pmax(dataset[,'baseHeight'],1)
dataset[,'diff.width.viewport'] <- abs(dataset[,'baseWidth'] - dataset[,'targetWidth']) / dataset[,'diff.viewport']
dataset[,'imageDiff.size'] <- dataset[,'imageDiff'] / pmax(dataset[,'size'], 1)
dataset[,'chiSquared.size'] <- dataset[,'chiSquared'] / pmax(dataset[,'size'], 1)
dataset[,'size'] <- pmax(dataset[,'baseHeight'] * dataset[,'baseWidth'], dataset[,'targetHeight'] * dataset[,'targetWidth'])

features_target <- c(
    'size',
#    'baseHeight', 'baseWidth', 'targetHeight', 'targetWidth',
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
    'childsNumber',
    'textLength'
)
dataset_train <- dataset

X_train_target <- dataset_train[,features_target]
y_train <- dataset_train[,'Result']

cost <- matrix(c(1,1,1,1), nrow=2, ncol=2, byrow=T)
dimnames(cost) <- list(c(0, 1), c(0, 1))

print(features_target)
m_target <- C5.0(X_train_target, y_train, trials=1, control=C5.0Control(winnow=T))

r <- predict(m_target, dataset_train[,features_target])
dataset_train[,'h'] <- r
errors <- subset(dataset_train, h != Result)
print(table(y_train, r))
