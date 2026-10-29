function mmalignBCtogene(bcidx,geneidx,bcfname,genefname)
%align barcode sequencing images. bcidx and geneidx indicate the channel in
%bc and gene image to be registered.
if ~exist('bcfname','var')
    bcfname='bc';
end

if ~exist('genefname','var')
    genefname='gene';
end



bcfiles=dir(['*',bcfname,'*.tif']);
genefiles=dir(['*',genefname,'*.tif']);
bcfiles=sort_nat({bcfiles.name});
genefiles=sort_nat({genefiles.name});

%read first cycle images
geneim=imread(genefiles{1},geneidx);
bcim=imread(bcfiles{1},bcidx);

%register images and find transformation
% [optimizer,metric] = imregconfig('multimodal');
% optimizer.InitialRadius = optimizer.InitialRadius/50;
% optimizer.GrowthFactor=1.01;
% optimizer.MaximumIterations=optimizer.MaximumIterations*10;

% tform = imregtform(bcim(300:end-300,300:end-300), geneim(300:end-300,300:end-300), ...
%     'translation', optimizer, metric,'PyramidLevels',4);
%tform = imregtform(bcim(end-500:end,1:500), geneim(end-500:end,1:500), 'translation', optimizer, metric);

tform=imregcorr(bcim(100:end-100,100:end-100), geneim(100:end-100,100:end-100), 'translation');


%%manually align pregfp to seq01
%[mpoints,fpoints]=cpselect(min((gfpim(:,:,gfpidx)/max(reshape(gfpim(:,:,gfpidx),[],1)))*2,1),min((seqim(:,:,seqidx)/max(reshape(seqim(:,:,seqidx),[],1)))*5,1),'Wait',true);
%tform=fitgeotrans(mpoints,fpoints,'nonreflectivesimilarity');


Rfixed=imref2d(size(sum(geneim,3)));
alignedbcim=uint16(imwarp(bcim,tform,'OutputView',Rfixed));
imwrite(double(geneim)./max(double(geneim(:))),'comp.tif');
imwrite(double(alignedbcim)./max(double(alignedbcim(:))),'comp.tif','WriteMode','Append');

%transform all barcode files
for i=1:length(bcfiles)
    bcinfo=imfinfo(bcfiles{i});
    im=zeros(bcinfo(1).Height, bcinfo(1).Width,size(bcinfo,1));
    for n=1:size(bcinfo,1)
        im(:,:,n)=imread(bcfiles{i},n);
    end
    alignedim=uint16(imwarp(im,tform,'OutputView',Rfixed));
    %figure;imshowpair(min((alignedgfpim(:,:,gfpidx)/max(reshape(gfpim(:,:,gfpidx),[],1)))*1,1),min((seqim(:,:,seqidx)/max(reshape(seqim(:,:,seqidx),[],1)))*3,1));
    imwrite(alignedim(:,:,1),['reg',bcfiles{i}]);
    for n=2:size(alignedim,3)
        imwrite(alignedim(:,:,n),['reg',bcfiles{i}],'WriteMode','Append');
    end
end