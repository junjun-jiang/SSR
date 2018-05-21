function [w]=fusedLeastR_My(A, y, lambda1, lambda2)

rho=lambda1;          % the regularization parameter
                    % it is a ratio between (0,1), if .rFlag=1

%----------------------- Set optional items ------------------------
opts=[];

% Starting point
opts.init=2;        % starting from a zero point

% termination criterion
opts.tFlag=5;       % run .maxIter iterations
opts.maxIter=2000;   % maximum number of iterations

% normalization
opts.nFlag=0;       % without normalization

% regularization
opts.rFlag=1;       % the input parameter 'rho' is a ratio in (0, 1)
%opts.rsL2=0.01;    % the squared two norm term

% fused penalty
opts.fusedPenalty=lambda2;

% line search
opts.lFlag=0;

%----------------------- Run the code LeastR -----------------------

[w, funVal1, ValueL1]= fusedLeastR(A, y, rho, opts);