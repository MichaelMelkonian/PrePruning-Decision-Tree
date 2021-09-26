#-------------------------------------------------------------------------
# AUTHOR: Michael Melkonian
# FILENAME: Assignment2_2
# SPECIFICATION: Here we will use a variation of 3 testing sets to observe the accuracy of our models predictions
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import tree
import csv


dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
   # X =
    age = {"Young":1,"Prepresbyopic":2, "Presbyopic":3} #Equating age value to 1,2, or 3
    speculate_prescrip = {"Myope":1, "Hypermetrope":2} #Equating SP value to 1 or 2
    astigmatism = {"No":1, "Yes":2} #Equating Astig value to 1 or 2
    tear_production_rate = {"Reduced":1, "Normal":2} #Equating TPR value to 1 or 2
    
    
    #transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    recommended_lenses = {"No":1, "Yes":2} #Equating CLASSES to 1 or 2
    # Create dictionary for mapping feature/class categories to numeric values
    for i in range(len(dbTraining)): #traversing through DB or csv file
        feature_row = [age.get(dbTraining[i][0]), speculate_prescrip.get(dbTraining[i][1]), astigmatism.get(dbTraining[i][2]), tear_production_rate.get(dbTraining[i][3])] #assigning value to 2d values to columns (age) (astigmatism)
        X.append(feature_row)
    
        class_row = recommended_lenses.get(dbTraining[i][4]) #assiging 2d values to classes (Recommended Lenses)
        Y.append(class_row)
    #loop your training and test tasks 10 times here
    for i in range (10):

       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       #--> add your Python code here
       #read the test data and add this data to dbTest
       dbTest = [] 
       with open('contact_lens_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           for j, row in enumerate(reader):
               if j > 0: #skipping the header
                   dbTest.append (row)
        #instantiating True Pos, True Neg, False Pos, False Neg values
       TP = 0
       TN = 0
       FP = 0
       FN = 0
       for data in dbTest:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
            dbX = []
            dbY = []
           
            ageTest = {"Young":1,"Prepresbyopic":2, "Presbyopic":3} #Equating age value to 1,2, or 3
            speculate_prescripTest = {"Myope":1, "Hypermetrope":2} #Equating SP value to 1 or 2
            astigmatismTest = {"No":1, "Yes":2} #Equating Astig value to 1 or 2
            tear_production_rateTest = {"Reduced":1, "Normal":2} #Equating TPR value to 1 or 2
            recommended_lensesTest = {"No":1, "Yes":2} #Equating CLASSES to 1 or 2
        
            feature_rowTest = [age.get(data[0]), speculate_prescripTest.get(data[1]), astigmatismTest.get(data[2]), tear_production_rateTest.get(data[3])] #assigning value to 2d values to columns (age) (astigmatism)
            dbX.append(feature_rowTest) #appending test csv values to secondary test db, "dbX"


        
            class_rowTest = recommended_lensesTest.get(data[4]) #assigning 2d values to classes (Recommended Lenses)
            dbY.append(class_rowTest) #appending test csv values to seconday test db for classes, "dbY"
            class_predicted = clf.predict(dbX)[0]
  

  
        #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
           #--> add your Python code here
            if(class_rowTest == class_predicted):
                if(class_rowTest == 2):
                    TP = TP + 1
                elif(class_rowTest == 1):
                    TN = TN + 1
            else:
                if (class_rowTest == 2):
                    FN = FN + 1
                if (class_rowTest == 1):
                    FP = FP + 1
            


            accPercent = (TP + TN) / (TP + TN + FP + FN)
   
        #find the lowest accuracy of this model during the 10 runs (training and test set)
        #--> add your Python code here
    temp = 1.00000
    for i in range(10):
        if(accPercent < temp):
            temp=accPercent
    
   

    #print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here

    print("Final accuracy when training on", ds, ":", temp)


