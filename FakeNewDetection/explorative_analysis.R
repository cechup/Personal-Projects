data <- read.csv("modelling_data1_wel.csv")
head
pca <- prcomp(as.matrix(data[,2:46]), scale. = T)
library(ggfortify)
autoplot(pca, loadings=T, loadings.label=T, data=data, col="label")

library(Rtsne)
set.seed(15)
tnse <- Rtsne(data, dims=2, perplexity = 5)
library(plotly)
points <- data.frame(x1=tnse$Y[,1],x2=tnse$Y[,2], label=data$label)
plot_ly(points, x=~x1,y=~x2,color=~person) %>% add_markers()
plot_ly(tsne)

X_nlp[50500,]

X_nlp <- as.matrix(data[,2:7])
head(X_nlp)
X_nlp <- scale(X_nlp)
X_bow <- as.matrix(data[,27:46])
X_em <- as.matrix(data[,28:37])
X_topic <- as.matrix(data[,38:45])
fa_nlp <- factanal(X_nlp, 2, scores = "Bartlett")
pca_nlp <- prcomp(X_nlp)
?factanal
fa_bow <- factanal(X_bow, 2, scores = "Bartlett")
fa_em <- factanal(X_em, 2, scores = "regression")
fa_topic <- factanal(X_topic, 2, scores = "regression")
autoplot(fa_nlp, col="label", data=data, loadings=T, loadings.label=T)
autoplot(fa_nlp, col="label", data=data, loadings=T, loadings.label=T)
autoplot(fa_bow, col="label", data=data, loadings=T, loadings.label=T)
autoplot(fa_em, col="label", data=data, loadings=T, loadings.label=T)
autoplot(fa_topic, col="label", data=data, loadings=T, loadings.label=T)

pca_nlp <- prcomp(X_nlp, scale. = T)
pca_bow <- prcomp(X_bow, scale. = T)
pca_em <- prcomp(X_em, scale. = T)
pca_topic <- prcomp(X_topic, scale. = T)
plot(pca_topic)
summary(pca_topic)
par(mfrow=c(1,1))
labels <- ifelse(data$label==1, "fake", "real")
data$label <- as.factor(labels)

cols = ifelse(data$label==1,"navy","deepskyblue")
cols <- c("fake" = "navy", "real" = "deepskyblue")
autoplot(pca_nlp, loadings=T, loadings.label=T, data=data,
         col="label")
autoplot(pca_bow, loadings=T, loadings.label=T, data=data, col="label")
autoplot(pca_em, loadings=T, loadings.label=T, data=data, col="label")
autoplot(pca_topic, loadings=T, loadings.label=T, data=data, col="label")
pca_nlp1 <- prcomp(as.matrix(data[,3:5]), scale. = T)
autoplot(pca_nlp1, loadings=T, loadings.label=T, data=data, col="label")

plot(data$perc7, data$perc11)
boxplot(as.factor(data$label),data$perc11)

barplot(prop.table(table(data$label[data$perc7>=0.1])), 
        xlab = "Real = 0, Fake = 1",main="Documenti con parole lunghe")

label <- ifelse(data$label=="FAKE",1,0)
boxplot(data$perc11 ~ data$label)
boxplot(data$perc7 ~ data$label, xlab="Real=0, Fake=1", 
        ylab = "parole con più di 11 caratteri")

barplot(prop.table(table(data$label[data$length<=250])), beside=T,
        main = "Testi inferiori a 250 parole", xlab = "0 = Real, 1 = Fake")

barplot(prop.table(rbind(table(data$label[data$length>1000]),
table(data$label[data$length>2000]),
table(data$label[data$length>3000]),
table(data$label[data$length>5000])),1), beside=T, 
legend.text = c(">1000", ">2000", ">3000", ">5000"),
main = "Frequenza di testi lunghi")



prop.table(table(data$label[data$length>=250]), beside=T)

table(table(data$label[data$length<250]), table(data$label[data$length>=250])

l <- cbind(length(data$label[data$length<250]), 
           length(data$label[data$length>1000]))      

boxplot(data$quotes ~ data$label)
barplot(prop.table(table(data$label[data$quotes>0])),
        xlab="Real = 0, Fake = 1", main="Frequenza di citazioni",
        col=4)

library(scatterplot3d)
scatterplot3d(pca_topic$x[,1],pca_topic$x[,2],pca_topic$x[,3],
              color=data$label+6,
              main = "Grafico 3 componenti pincipali",
              xlab = "Prima componente", ylab = "Seconda componente",
              zlab = "Terza componente", pch=1)
legend("topleft", legend=c("real news","fake news"),col=c(6,7),
       pch=1)

scatterplot3d(pca_em$x[,1],pca_em$x[,2],pca_em$x[,3],
              color=as.numeric(data$label)+5,
              main = "Grafico 3 componenti pincipali",
              xlab = "Prima componente", ylab = "Seconda componente",
              zlab = "Terza componente", pch=1)
legend("topleft", legend=c("real news","fake news"),col=c(6,7),
       pch=1)


########## T-SNE
library(Rtsne)
library(tidyverse)
set.seed(15)
X <- as.matrix(data[,-1])
y <- data[,1]
Xs <- scale(X)
tnse <- Rtsne(Xs,dims=2,perplexity = 5,check_duplicates = FALSE)
?Rtsne
library(plotly)
points <- data.frame(x1=tnse$Y[1,],x2=tnse$Y[2,],label=y)
plot_ly(points, x=~x1,y=~x2) %>% add_markers()
