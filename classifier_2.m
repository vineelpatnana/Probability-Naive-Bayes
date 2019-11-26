function [P,I] = classifier_2(input1,input2,mean_vals1,var_vals1,mean_vals2,var_vals2)

n_i_1 = length(input1(:,1));%finding number of observations
n_c_1 = length(mean_vals1(1,:));%finding number of classes

for i = 1:n_i_1
  for j = 1:n_c_1
      for k = 1:n_c_1
        %finding the z scores to futher find the probabilty
        z_1(j,k) = (input1(i,j) - mean_vals1(1,k))/(sqrt(var_vals1(1,k)));
        z_2(j,k) = (input2(i,j) - mean_vals2(1,k))/(sqrt(var_vals2(1,k)));
      end
      %finding the P(Ci/F) = (P(F/Ci)*P(Ci)/P(F)) 
      p_1 = normpdf(z_1(j,:));
      p_2 = normpdf(z_2(j,:));
      %taking the max value index after the multiplication of the two probabilities obtained
      %which in turn gives the predicted class
      [P(i,j),I(i,j)] = max(p_1.*p_2);
  end
end
end
