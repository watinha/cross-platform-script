featureScale <- function (dataset) {
    scaled_dataset <- data.frame(dataset)
    for (i in 1:(ncol(dataset))) {
        if (is.numeric(dataset[,i])) {
            scaled_dataset[,i] <- scale(dataset[,i])
        } else {
            scaled_dataset[,i] <- dataset[,i]
        }
    }
    return (scaled_dataset)
}
