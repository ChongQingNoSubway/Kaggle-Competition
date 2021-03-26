clear
clc

load('data.mat')
N=size(y,1);
k=3;
N_train=floor(0.7*(N-k)); %%number of the traning dataset
train_label=randperm(N-k,N_train); %%index of the traning dataset
x_train=zeros(N_train,3); %% define the size of the trainning dataset
y_train=zeros(N_train,1); %% define the size of the trainning dataset
x_test=zeros(N-k-N_train,3); %% define the size of the testing dataset
y_test=zeros(N-k-N_train,1); %% define the size of the testing dataset
o1=1;
o2=1;
for i=1:N-k
    if ismember(i, train_label) %% define the training of polynomial feature (II in the report) 
        x_train(o1,1)=y(i);   
        x_train(o1,2)=y(i+1);
        x_train(o1,3)=y(i+2);
        y_train(o1,1)=y(i+3);
        o1=o1+1;
    else                       %% define the Testing of polynoimal feature
        x_test(o2,1)=y(i);
        x_test(o2,2)=y(i+1);
        x_test(o2,3)=y(i+2);
        y_test(o2,1)=y(i+3);
        o2=o2+1;
    end
end

%%
w=pinv(x_train' * x_train)* x_train'* y_train; %% optimaze the W
y_train_pred=x_train*w;   %% predict the training y
y_test_pred=x_test*w;     %% predict the testing y
error_train=mean((y_train-y_train_pred).^2); %% caculate the MSE of the training
error_test=mean((y_test-y_test_pred).^2);   %% caculate the MSE of the testing