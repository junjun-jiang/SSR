%example_mat_primal

clear all;
clc

% add path
addpath(genpath('../../SLEP/'));

% load data and set regularization parameter
load('../../data/scene.mat');
lambda = 10^-4;

% center data
D = CenterRowData(D);
L = CenterRowData(L);

% call the main function
[W] = mat_primal(D,L,lambda); 







