setwd("~/gdrive/REU/imbalance_algorithms/dataset/")
df<-read.table("abalone19.dat",header = FALSE,skip = 13,sep = ",")
x=df$V1
x <- as.factor(x)
levels(x) <- 1:length(levels(x))
x <- as.numeric(x)
df$V1=x
x=df$V3
x <- as.factor(x)
levels(x) <- 1:length(levels(x))
x <- as.numeric(x)
df$V3=x
x=df$V4
x <- as.factor(x)
levels(x) <- 1:length(levels(x))
x <- as.numeric(x)
df$V4=x
names(df)[length(df)]="Class"
names(df)
nrow(df)
head(df,n=5)
x<-split(df,f=df$Class)
df0<-as.data.frame(x$negative)
df1<-as.data.frame(x$positive)
nrow(df0)/nrow(df1)
length(names(df))
write.table(df,file="abalone19.dat",sep = ",")
