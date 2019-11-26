clc
clear 
close all

%loading data from the data file provided
load('data.mat')

% Step 1 : Training: Use the data of the first 100 subjects to estimate
% mean and variance of F1 and F2

% creating training and test set from F1
f1_train = F1(1:100,:);
f1_test = F1(101:1000,:);

% creating training and test set from F2
f2_train = F2(1:100,:);
f2_test = F2(101:1000,:);

% calculating mean and variance from training data of F1
mean_vals(1,:) = mean(f1_train,1);
var_vals(1,:) = var(f1_train,1);

% calculating mean and variance from training data of F2
mean_vals(2,:) = mean(f2_train,1);
var_vals(2,:) = var(f2_train,1);

% Step 2 : Testing: Assume that X = F1. Using the Bayes' theorem, 
% calculate the probability of each class for data of the remaining 
% subjects (columns 101-1000 of F1) and consequently predict the class
% for each data point. 
% Note that each subject performed 5 different tasks so you need to predict 
% the class of 4500 data points.

% Step 2.2.Calculating the accuracy of the classifier: You need to check 
% the percentage of the data whose class are correctly predicted. 
% Classification accuracy = correct predictions / total predictions
% Error rate = incorrect predictions / total predictions

%Initializing the actual matrix for comparison with the predictions
actual = zeros(900, 5);
actual(:,1) = 1;
actual(:,2) = 2;
actual(:,3) = 3;
actual(:,4) = 4;
actual(:,5) = 5;

%Using claffifier1 function to find the probabilities and thus the class
%prediction of each data point

[z_F1,p_F1,I_F1] = classifier1(f1_test,mean_vals(1,:),var_vals(1,:));

error_mat_F1 = actual - I_F1;% find errors between actual and predicted
idx_F1=error_mat_F1==0; %returns 1 at position where value is zero and zeros everywhere else
error_F1=sum(idx_F1(:));%finding the number of data pointss correctly classfied 

Correction_Rate_F1 = error_F1/4500; %0.5262
Error_Rate_F1 = (4500-error_F1)/4500; %0.4738


% Step 3: Standard Normal (Z-Score): Assume F1 to be a subjective measure. 
% In this case the mean value and the range of data reported by one subject 
% will not be consistent with another subject. In other to remove the effect 
% of individual differences, you have to normalize the data of each subject 
% using the standard normal formulation (removing the mean and dividing by 
% standard deviation). Calculate F1 (the standard normal of F1) and plot the 
% distribution of the data using F1 and F2, and compare it to the 
% distribution in F1 and F2 shown on right.

%Calculating Z1
for i = 1:1000
Z1(i,:) = zscore(F1(i,:));
end

%Caluclating Z2
for i = 1:1000
Z2(i,:) = zscore(F2(i,:));
end

% Plot Z-scores

hold on
for i = 1:5
scatter(Z1(:,i),F2(:,i))
hold on
end
title('Scatterplot: Normalized Features')
xlabel('1^{st} Feature (Z1)')
ylabel('2^{nd} Feature (F2)')
legend('C1','C2','C3','C4','C5')
hold off

% Step 3
%Case 2: X = Z1

% creating training and test set from Z1
Z1_train = Z1(1:100,:);
Z1_test = Z1(101:1000,:);

% calculating mean and variance from training data of Z1
mean_vals_Z1(1,:) = mean(Z1_train,1);
var_vals_Z1(1,:) = var(Z1_train,1);

%Using claffifier1 function to find the probabilities and thus the class
%prediction of each data point
[z_Z1,p_Z1,I_Z1] = classifier1(Z1_test,mean_vals_Z1(1,:),var_vals_Z1(1,:));

error_mat_Z1 = actual - I_Z1;% find errors between actual and predicted
idx_Z1=error_mat_Z1==0; %returns 1 at position where value is zero and zeros everywhere else
error_Z1=sum(idx_Z1(:));%finding the number of data pointss correctly classfied 

Correction_Rate_Z1 = error_Z1/4500;%0.8838
Error_Rate_Z1 = (4500-error_Z1)/4500;%0.1162

%Case 3: X = F2

%Using claffifier1 function to find the probabilities and thus the class
%prediction of each data point
[z_F2,p_F2,I_F2] = classifier1(f2_test,mean_vals(2,:),var_vals(2,:));

error_mat_F2 = actual - I_F2;% find errors between actual and predicted
idx_F2=error_mat_F2==0; %returns 1 at position where value is zero and zeros everywhere else
error_F2=sum(idx_F2(:));%finding the number of data pointss correctly classfied 

Correction_Rate_F2 = error_F2/4500; %0.5351
Error_Rate_F2 = (4500-error_F2)/4500;%0.4649

%Case 4: X = [Z1 F2]
%Multivariate case

%Using claffifier_2 function to find the probabilities and thus the class
%prediction of each data point
[P,I] = classifier_2(Z1_test,f2_test,mean_vals_Z1(1,:),var_vals_Z1(1,:),mean_vals(2,:),var_vals(2,:));

error_mat = actual - I;% find errors between actual and predicted
idx=error_mat==0; %returns 1 at position where value is zero and zeros everywhere else
error=sum(idx(:));%finding the number of data pointss correctly classfied 

Correction_Rate = error/4500; %0.9784
Error_Rate = (4500-error)/4500;%0.1162

% Plotting F1 and F2 to derive inferences of their correlation
%F1 and F2
for i = 1:5
scatter(F1(:,i),F2(:,i))
hold on
end
title('Scatterplot')
xlabel('1^{st} Feature (F1)')
ylabel('2^{nd} Feature (F2)')
legend('C1','C2','C3','C4','C5')
hold off

% Plotting Z1 and Z2 to derive inferences of their correlation
%Z1 and Z2
for i = 1:5
scatter(Z1(:,i),Z2(:,i))
hold on
end
title('Scatterplot')
xlabel('1^{st} Feature (Z1)')
ylabel('2^{nd} Feature (Z2)')
legend('C1','C2','C3','C4','C5')
hold off
