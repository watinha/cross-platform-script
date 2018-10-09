metrics <- function (confusion_table) {
    tp <- confusion_table[2,2]
    tn <- confusion_table[1,1]
    fp <- confusion_table[1,2]
    fn <- confusion_table[2,1]
    precision <- tp / (tp + fp)
    recall <- tp / (tp + fn)
    Fscore <- 2 * ((precision * recall) / (precision + recall))
    if (is.nan(precision)) {
        precision <- 0
    }
    if (is.nan(recall)) {
        recall <- 0
    }
    if (is.nan(Fscore)) {
        Fscore <- 0
    }
    return (list(confusion_matrix=confusion_table, precision=precision, recall=recall, Fscore=Fscore))
}

crossValidation <- function (dataset_cv, dataset_test, folds, target_features, call, table_name) {
    # Setting output
    csv <- matrix(nrow = 14, # 1 column header + 1 trainning result + 10 fold cross-validation result
                  ncol = 8) # 1 column for title + 7 metrics

    csv[1, 2] <- 'TP'
    csv[1, 3] <- 'TN'
    csv[1, 4] <- 'FP'
    csv[1, 5] <- 'FN'
    csv[1, 6] <- 'Precision'
    csv[1, 7] <- 'Recall'
    csv[1, 8] <- 'F-Score'

    # TRAINNING RESULTS
    csv[2, 1] <- 'Trainning metrics'
    # Target approach
    X <- dataset_cv[,target_features[1:(length(target_features) - 1)]]
    y <- dataset_cv[,'Result']
    train_results_target <- call(X, y, X, y)

    # SAVING RESULTS
    r <- metrics(train_results_target)
    csv[2, 2] <- r[['confusion_matrix']][2, 2]
    csv[2, 3] <- r[['confusion_matrix']][1, 1]
    csv[2, 4] <- r[['confusion_matrix']][1, 2]
    csv[2, 5] <- r[['confusion_matrix']][2, 1]
    csv[2, 6] <- r[['precision']]
    csv[2, 7] <- r[['recall']]
    csv[2, 8] <- r[['Fscore']]


    # CROSS-VALIDATION
    names <- names(folds)
    for (i in 1:length(names)) {
        train_fold <- dataset_cv[-folds[[names[i]]],]
        cv_fold <- dataset_cv[folds[[names[i]]],]
        # Target approach
        X <- train_fold[,target_features[1:(length(target_features) - 1)]]
        y <- train_fold[,'Result']
        X_cv <- cv_fold[,target_features[1:(length(target_features) - 1)]]
        y_cv <- cv_fold[,'Result']
        results_target <- call(X, y, X_cv, y_cv)
        # Saving results
        csv[2 + i, 1] <- 'Fold'
        r <- metrics(results_target)
        csv[2 + i, 2] <- r[['confusion_matrix']][2, 2]
        csv[2 + i, 3] <- r[['confusion_matrix']][1, 1]
        csv[2 + i, 4] <- r[['confusion_matrix']][1, 2]
        csv[2 + i, 5] <- r[['confusion_matrix']][2, 1]
        csv[2 + i, 6] <- r[['precision']]
        csv[2 + i, 7] <- r[['recall']]
        csv[2 + i, 8] <- r[['Fscore']]
    }
    # CV total
    csv[13, 1] <- 'CV Avg'
    csv[13, 2] <- mean(as.integer(csv[3:12, 2]))
    csv[13, 3] <- mean(as.integer(csv[3:12, 3]))
    csv[13, 4] <- mean(as.integer(csv[3:12, 4]))
    csv[13, 5] <- mean(as.integer(csv[3:12, 5]))
    csv[13, 6] <- mean(as.double(csv[3:12, 6]))
    csv[13, 7] <- mean(as.double(csv[3:12, 7]))
    csv[13, 8] <- mean(as.double(csv[3:12, 8]))

    # TEST RESULTS
    # Target approach
    X_test <- dataset_test[,target_features[1:(length(target_features) - 1)]]
    y_test <- dataset_test[,'Result']
    test_results_target <- call(X, y, X_test, y_test)
    # Saving results
    csv[14, 1] <- 'Test'
    r <- metrics(test_results_target)
    csv[14, 2] <- r[['confusion_matrix']][2, 2]
    csv[14, 3] <- r[['confusion_matrix']][1, 1]
    csv[14, 4] <- r[['confusion_matrix']][1, 2]
    csv[14, 5] <- r[['confusion_matrix']][2, 1]
    csv[14, 6] <- r[['precision']]
    csv[14, 7] <- r[['recall']]
    csv[14, 8] <- r[['Fscore']]

    write.table(csv, file=table_name, sep=",", qmethod='double', row.names=FALSE, col.names=FALSE)
    return (csv)
}
