function cellid1=mmassignrol2cell(lroi1,imcell1)
%assign rolonies to cells. 
cellid1=double(imcell1(sub2ind(size(imcell1),lroi1(:,2),lroi1(:,1))));%this allows for 9999 cells per tile. Should be enough.
