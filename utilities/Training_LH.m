function [YH YL] = Training_LH(upscale,BlurWindow,nTraining)
%%% construct the HR and LR training pairs from the FEI face database
disp('Constructing the HR-LR training set...');
for i=1:nTraining
    %%% read the HR face images from the HR training set
    strh = strcat('.\trainingFaces\',num2str(i),'_h.jpg');    
    HI = double(imread(strh)); 
    YH(:,:,i) = HI;
    
    %%% generate the LR face image by smooth and down-sampling
    w=fspecial('average',[BlurWindow BlurWindow]);
    SI = imfilter(HI,w);
    LI = imresize(SI,1/upscale,'bicubic');
    YL(:,:,i) = LI;
    strL = strcat('.\trainingFaces\',num2str(i),'_l.jpg');
    imwrite(uint8(LI),strL,'jpg'); 
end

disp('done.');