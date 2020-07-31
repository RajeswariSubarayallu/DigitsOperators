# DigitsOperators
Dataset model for both digits and operators combined.
Total no.of classes are 14
Labels are defined as follows:
0->0
1->1
2->2
3->3
4->4
5->5
6->6
7->7
8->8
9->9
10->+
11->-
12->*
13->/
handwritten_classifier.py contains code for tarining the dataset and to freeze the model.
All images are of size 28*28 with black background.
Operator.h5 is a keras model for digits and operators.
data.pb is a protopuf file to be used for android applications.
