
function pos=stitch_10x_images_mist(fname,reg_ch,overlap,rescale_factor,mock)
% stitch images using ImageJ and save tforms. Requires imageJ installation
% If mock, will try to piece together image using an approximation of the
% stitched positions based on 25% overlap. Only works with Kinetix
% (3200x3200).
% This version works using MIST in ImageJ





if ~exist('mock','var')
    mock=0;
end
fprintf('Stitching whole-slice image.\n');
if mock~=1
    niestitch_mist(fname,reg_ch,overlap); % This doesn't work reliably with DIC channel, adjust imaging settings.
    % read mist results
    read_mist_results(fname,rescale_factor)
else
  
    xcam=3200;
    ycam=3200;

    xinterval=xcam*(1-overlap)*rescale_factor;
    yinterval=ycam*(1-overlap)*rescale_factor;

    tform40xto10x={};
    [folders,pos,xxx,yyy]=get_folders();
    for n=1:numel(folders)
        xt=(max(xxx)-xxx(n))*xinterval;
        yt=yyy(n)*yinterval;
        T=[0.5,0,0;0,0.5,0;xt,yt,1];
        tform40xto10x{n}=affine2d(T);
    end
    save('40xto10x.mat','tform40xto10x');
    
    fprintf('All done. Saved transformations to 40xto10x.mat.\n')
end


end



