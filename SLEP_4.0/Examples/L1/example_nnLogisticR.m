clear, clc;

% This is an example for running the function nnLogisticR
%
%  Problem:
%
%  min  f(x,c) = - weight_i * log (p_i) + 1/2 * rsL2 * ||x||_2^2 
%                 + rho * \|x\|_1 
%
%  a_i denotes a training sample,
%      and a_i' corresponds to the i-th row of the data matrix A
%
%  y_i (either 1 or -1) is the response
%     
%  p_i= 1/ (1+ exp(-y_i (x' * a_i + c) ) ) denotes the probability
%
%  weight_i denotes the weight for the i-th sample
%
% For detailed description of the function, please refer to the Manual.
%
%% Related papers
%
% [1]  Jun Liu and Jieping Ye, Efficient Euclidean Projections
%      in Linear Time, ICML 2009.
%
% [2]  Jun Liu and Jieping Ye, Sparse Learning with Efficient Euclidean
%      Projections onto the L1 Ball, Technical Report ASU, 2008.
%
% [3]  Jun Liu, Jianhui Chen, and Jieping Ye, 
%      Large-Scale Sparse Logistic Regression, KDD, 2009.
%
%% ------------   History --------------------
%
% First version on August 10, 2009.
%
% September 5, 2009: adaptive line search is added
%
% For any problem, please contact Jun Liu (j.liu@asu.edu)

cd ..
cd ..

root=cd;
addpath(genpath([root '/SLEP']));
                     % add the functions in the folder SLEP to the path
                   
% change to the original folder
cd Examples/L1;

m=1000;  n=1000;     % The data matrix is of size m x n

randNum=1;           % a random number

% ---------------------- generate random data ----------------------
randn('state',(randNum-1)*3+1);
A=randn(m,n);        % the data matrix

randn('state',(randNum-1)*3+2);
xOrin=randn(n,1);

randn('state',(randNum-1)*3+3);
y=[ones(n/2,1);...
    -ones(n/2, 1)];  % the response

rho=0.1;             % the regularization parameter
                     % it is a ratio between (0,1), if .rFlag=1

%----------------------- Set optional items -----------------------
opts=[];

% Starting point
opts.init=2;        % starting from a zero point

% Termination 
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=40;    % maximum number of iterations

% Normalization
opts.nFlag=0;       % without normalization

% Regularization
opts.rFlag=1;       % the input parameter 'rho' is a ratio in (0, 1)
opts.rsL2=0;        % the squared two norm term

% Group Property
opts.sWeight=[1,2]; % set the weight for positive and negative samples

%----------------------- Run the code LogisticR -----------------------
[x, c, funVal]=nnLogisticR(A, y, rho, opts);

figure;
plot(funVal);
xlabel('Iteration (i)');
ylabel('The objective function value');

% --------------------- compute the pathwise solutions ----------------
opts.fName='nnLogisticR';    % set the function name to 'LogisticR'
Z=[0.5, 0.2, 0.1, 0.01];   % set the parameters

% run the function pathSolutionLogistic
fprintf('\n Compute the pathwise solutions, please wait...');
[X,C]=pathSolutionLogistic(A, y, Z, opts);