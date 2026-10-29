function contours=mmsetsurface(slicenames,idx)
%select points delineating the top and bottom of the cortex. idx indicates
%the channel to display.
contours=struct;
%files=dir('aligned*.tif');
files=dir('*.tif');
files=sort_nat({files.name});
for n=1:length(slicenames)
    im=imread(files{n},idx);
    im=padarray(im,[500,500],'both');
    im1=sort(im(:),'descend');
    figure; imshow(im,[0 prctile(im1(:),99.5)],'InitialMagnification',50);title([slicenames{n},', set top, press enter; then set bottom']);
    hold on;
    topx=[];topy=[];
    while 1
        [x,y]=myginput(1,'topl');
        if numel(x)==1
            topx=[topx;x];
            topy=[topy;y];
            if numel(topx)>=2
                plot(topx(end-1:end),topy(end-1:end),'r','LineWidth',2);
            else
                scatter(x,y,10,'r','filled');
            end
        else
            break
        end
    end
    bottomx=[];bottomy=[];
    while 1
        [x,y]=myginput(1,'cross');
        if numel(x)==1
            bottomx=[bottomx;x];
            bottomy=[bottomy;y];
            if numel(bottomx)>=2
                plot(bottomx(end-1:end),bottomy(end-1:end),'g','LineWidth',2);
            else
                scatter(x,y,10,'g','filled');
            end
        else
            break
        end
    end
   
    close all;
    contours(n).slice=slicenames{n};
    contours(n).top=[topx,topy]-500;
    contours(n).bottom=[bottomx,bottomy]-500;
end
save('contours.mat','contours');


