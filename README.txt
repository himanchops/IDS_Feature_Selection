INTRUSION DETECTION SYSTEM USING FEATURE SELECTION

Provided Dataset is first discretized using ROSETTA.
Select data file in ROSETTA interface; Select project, discretization algorithm - NaiveScaler for getting cuts and run the program to get discretized dataset;
Run normalization manually.

Code script is run using python 2.7 with mentioned libraries.
Input discretized dataset file name as system argument to code file.
Script reads and fits data into classes and features, create training, testing and validation sets;
Script creates base individuals and toolbox to register required functions for implementing algorithm. Parameters of the algorithm can be adjusted therein.
main function launches processing of the dataset with all features and calculates accuracy and time; Followed by launching genetic algorithm to get perfermance of all generation populations.
List of subsets is chosen which perform best on validation set (Hall of Fame objects);
Feature ranking is done on basis of performance of each feature over all generation.
Subset is chosen which performs best on vaidation set and displayed.


# Refer ROSETTA manual, DEAP manual page for individual functions.
Future work should involve optimizing fitness function and implementing parallelism for GA as it inherently supports individual processing.
