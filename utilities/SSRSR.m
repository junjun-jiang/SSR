
function im_SR = SSRSR(im_l,YH,YL,upscale,patch_size,overlap,lambda1,lambda2)

[imrow imcol nTraining] = size(YH);

Img_SUM      = zeros(imrow,imcol);
overlap_FLAG = zeros(imrow,imcol);

U = ceil((imrow-overlap)/(patch_size-overlap));  
V = ceil((imcol-overlap)/(patch_size-overlap)); 

% hallucinate the HR image patch by patch
for i = 1:U
    fprintf('.');
   for j = 1:V     

        BlockSize  =  GetCurrentBlockSize(imrow,imcol,patch_size,overlap,i,j);   
        if size(YH,1)==size(YL,1)*upscale
            BlockSizeS =  GetCurrentBlockSize(imrow/upscale,imcol/upscale,patch_size/upscale,overlap/upscale,i,j);  
        elseif size(YH,1)==size(YL,1)
            BlockSizeS =  BlockSize;
        end
        
        im_l_patch =  im_l(BlockSizeS(1):BlockSizeS(2),BlockSizeS(3):BlockSizeS(4));           % extract the patch at position£¨i,j£©of the input LR face     
        im_l_patch =  double(im_l_patch(:));   % Reshape 2D image patch into 1D column vectors   
        
        XF   =  Reshape3D(YH,BlockSize);    % reshape each patch of HR face image to one column
        X    =  Reshape3D(YL,BlockSizeS);   % reshape each patch of LR face image to one column

        % represent the LR patch at  position£¨i,j£©using our LcR 
        nframe =  size(im_l_patch',1);
        nbase  =  size(X',1);
        XX     =  sum(im_l_patch'.*im_l_patch', 2);        
        SX     =  sum(X'.*X', 2);
        D      =  repmat(XX, 1, nbase)-2*im_l_patch'*X+repmat(SX', nframe, 1); 
        [val indx] = sort(D);
        XF     = XF(:,indx);
        X      = X(:,indx);        
     
        [w]= fusedLeastR_My(X, im_l_patch, lambda1, lambda2);        
       
        % obtain the HR patch with the same weight vector w
        Img  =  XF*w; 
        
        % integrate all the LR patch        
        Img  =  reshape(Img,patch_size,patch_size);
        Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))      = Img_SUM(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))+Img;
        overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4)) = overlap_FLAG(BlockSize(1):BlockSize(2),BlockSize(3):BlockSize(4))+1;
    end
end
%  averaging pixel values in the overlapping regions
im_SR = Img_SUM./overlap_FLAG;
