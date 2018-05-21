% =========================================================================
% Face Hallucination via SSR
% Example code
%
% Reference
% J. Jiang, J. Ma, C. Chen, X. Jiang, and Z. Wang, ¡°Noise Robust Face ImageSuper-resolution through Smooth Sparse Representation,¡± IEEE Transactions on Cybernetics,DOI: 10.1109/TCYB.2016.2594184, 2016. (in press)
%
% For any questions, email me by junjun0595@163.com or jiangjunjun@nii.ac.jp
%=========================================================================

clc;close all;
clear all;
cd;addpath(genpath(cd));

% set parameters
nrow        = 120;        % rows of HR face image
ncol        = 100;        % cols of LR face image
nTraining   = 360;        % number of training sample
nTesting    = 40;         % number of ptest sample
upscale     = 4;          % upscaling factor 
BlurWindow  = 4;          % size of an averaging filter 
lambda1     = 0.0001;       % regularization parameters
lambda2     = 0;       
patch_size  = 16;         % image patch size
overlap     = 12;          % the overlap between neighborhood patches

% construct the HR and LR training pairs from the FEI face database
% [YH YL] = Training_LH(upscale,BlurWindow,nTraining);
load('fei.mat','YH','YL');
YL = imresize(YL,upscale);
fprintf('\nface hallucinating for %d input test images\n', nTesting);

for TestImgIndex = 1:nTesting

    fprintf('\nProcessing  %d/%d LR image\n', TestImgIndex,nTesting);

    % read ground truth of one test face image
    strh    = strcat('.\testFaces\',num2str(TestImgIndex),'_test.jpg');
    im_h    = imread(strh);

    % generate the input LR face image by smooth and down-sampleing
    w       = fspecial('average',[BlurWindow BlurWindow]);
    im_s    = imfilter(im_h,w);
    im_l    = imresize(im_s,1/upscale,'bicubic');
    im_l    = imresize(im_l,upscale,'bicubic');
    im_l    = double(im_l);
%     figure,imshow(im_l);title('input LR face');

    % add noise to the LR face image (Optional)
    v    =  0;seed   =  0;randn( 'state', seed );
    noise      =   randn(size(im_l));
    noise      =   noise/sqrt(mean2(noise.^2));  
    im_l       =   double(im_l) + v*noise;   
    im_l       =   double(im_l);  
    
    % face hallucination via LcR
    [im_SR] = SSRSR(im_l,YH,YL,upscale,patch_size,overlap,lambda1,lambda2);
    [im_SR] = uint8(im_SR);

    % bicubic interpolation for reference
    im_b = imresize(im_l, [nrow, ncol], 'bicubic');

    % compute PSNR and SSIM for Bicubic and our method
    bb_psnr(TestImgIndex) = psnr(im_b,im_h);
    bb_ssim(TestImgIndex) = ssim(im_b,im_h);

    sr_psnr(TestImgIndex) = psnr(im_SR,im_h);
    sr_ssim(TestImgIndex) = ssim(im_SR,im_h);

    % display the objective results (PSNR and SSIM)
    fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr(TestImgIndex));
    fprintf('PSNR for LcR Recovery: %f dB\n', sr_psnr(TestImgIndex));
    fprintf('SSIM for Bicubic Interpolation: %f dB\n', bb_ssim(TestImgIndex));
    fprintf('SSIM for LcR Recovery: %f dB\n', sr_ssim(TestImgIndex));

    % show the images
%     figure, imshow(im_b);
%     title('Bicubic Interpolation');
%     figure, imshow(uint8(im_SR));
%     title('LcR Recovery');
    
    % save the result
    strw = strcat('./results/',num2str(TestImgIndex),'_SR.bmp');
    imwrite(uint8(im_SR),strw,'bmp');
end

fprintf('===============================================\n');
fprintf('Average PSNR of Bicubic Interpolation: %f\n', sum(bb_psnr)/nTesting);
fprintf('Average PSNR of LcR method: %f\n', sum(sr_psnr)/nTesting);
fprintf('Average SSIM of Bicubic Interpolation: %f\n', sum(bb_ssim)/nTesting);
fprintf('Average SSIM of LcR method: %f\n', sum(sr_ssim)/nTesting);
fprintf('===============================================\n');