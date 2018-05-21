clear, clc;

% This is an example for running the function fusedLogisticR
% 
%% Problem
%
%  min  min  f(x,c) = - weight_i * log (p_i) + rho * ||x||_1 + 
%       opts.fusedPenalty * sum_i |x_i-x_{i+1}|
%
% For detailed description of the function, please refer to the Manual.
%
%% Related papers
%
% [1]  Jun Liu, Lei Yuan, and Jieping Ye, An Efficient Algorithm for 
%      a Class of Fused Lasso Problems, KDD, 2010.
%
%% ------------   History --------------------
%
% First version on 19 September 2010.
%
% For any problem, please contact Jun Liu (j.liu@asu.edu)

cd ..
cd ..

root=cd;
addpath(genpath([root '/SLEP']));
                   % add the functions in the folder SLEP to the path
                   
% change to the original folder
cd Examples/fusedLasso;      

m=100;  n=1000;    % The data matrix is of size m x n

% for reproducibility
randNum=1;

% ---------------------- Generate random data ----------------------
randn('state',(randNum-1)*3+1);
A=randn(m,n);       % the data matrix

y=[ones(ceil(m/2),1); -ones(m-ceil(m/2),1)];

rho=0.001;          % the regularization parameter
                    % it is a ratio between (0,1), if .rFlag=1

%----------------------- Set optional items ------------------------
opts=[];

% Starting point
opts.init=2;        % starting from a zero point

% termination criterion
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=500;   % maximum number of iterations

% normalization
opts.nFlag=0;       % without normalization

% regularization
opts.rFlag=1;       % the input parameter 'rho' is a ratio in (0, 1)
%opts.rsL2=0.01;    % the squared two norm term

% fused penalty
opts.fusedPenalty=0.01;

% line search
opts.lFlag=0;

%----------------------- Run the code LeastR -----------------------
tic;
[x1, c1, funVal1, ValueL1]= fusedLogisticR(A, y, rho, opts);
toc;

figure;
plot(funVal1,'-.b');
xlabel('Iteration (i)');
ylabel('The objective function value');

