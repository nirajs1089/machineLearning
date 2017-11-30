
#importing dataset
dataset = read.csv('Salary_Data.csv')
#dataset = dataset[,2:3]

#split the data into training and test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary,SplitRatio = 2/3)
training_set= subset(dataset,split == TRUE)
test_set= subset(dataset,split == FALSE)

# #feature scaling
# #all values in dataset must be numeric (as datatype not just values)
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])

#fitting simple linear regression to the training set
regressor = lm(formula = Salary ~  YearsExperience,data = training_set)

#predict the test set results
y_pred = predict(regressor,newdata = test_set)

#visualizing the training set results
#install.packages('ggplot2')
library(ggplot2)

ggplot() +
  geom_point(aes(x=training_set$YearsExperience,y=training_set$Salary),
             colour='red') + 
  geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,newdata = training_set)),
            colour='blue') +
  ggtitle('Salary vs Experience(Training Set)')
  xlab('Years of Experience') +
  ylab('Salary')
  
  ggplot() +
    geom_point(aes(x=test_set$YearsExperience,y=test_set$Salary),
               colour='red') + 
    geom_line(aes(x=training_set$YearsExperience,y=predict(regressor,newdata = training_set)),
              colour='blue') +
    ggtitle('Salary vs Experience(Test Set)')
    xlab('Years of Experience') +
    ylab('Salary')

