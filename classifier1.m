function [z,p,I] = classifier1(input,mean_vals,var_vals)

n_i = length(input(:,1));%finding number of observations
n_c = length(mean_vals(1,:));%finding number of classes

for i = 1:n_i
  for j = 1:n_c
      for k = 1:n_c
        %finding the z scores to futher find the probabilty 
        z(j,k) = (input(i,j) - mean_vals(1,k))/(sqrt(var_vals(1,k)));
      end
      %finding the P(Ci/F) = (P(F/Ci)*P(Ci)/P(F)) and taking th4e max
      %value index which in turn gives the predicted class
      [p(i,j),I(i,j)] = max(normpdf(z(j,:))/5); 
  end
end
end
