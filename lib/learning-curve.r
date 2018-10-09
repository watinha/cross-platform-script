source('lib/cross-validation.r')
learningCurve <- function (dataset_learning, dataset_cv, features, model_call) {
    folds <- createFolds(dataset_learning[,'Result'], k=100)
    index <- 0
    error_trainning <- c()
    error_cv <- c()
    sub_dataset <- NULL

    for (i in folds) {
        index <- index + 1
        if (is.null(sub_dataset)) {
            sub_dataset <- dataset_learning[i,]
        } else {
            sub_dataset <- rbind(sub_dataset, dataset_learning[i,])
        }
        sub_folds <- createFolds(sub_dataset[,'Result'], k=10)
        file_name <- paste('output/', index, '.csv', sep='')
        r <- crossValidation(sub_dataset, dataset_cv, sub_folds, features, model_call, file_name)
        error_trainning[index] <- (as.double(r[2, 4]) + as.double(r[2, 5])) / nrow(sub_dataset)
        error_cv[index] <- ((as.double(r[14, 4]) + as.double(r[14, 5])) * 10) / (nrow(dataset_cv) * 10)
    }
    result <- data.frame(error_trainning, error_cv)
    plot(1:100, error_cv, type='o', col='blue', xlim=c(1, 100), ylim=c(0, max(error_trainning, error_cv)))
    lines(1:100, error_trainning, type='o', col='red')
    return (result)
}
