
function checkregistration(fname,tformfname,scaling)
[folders,pos]=get_folders();
if ~exist('scaling','var')
    scaling=1;
end

if ~exist('tformfname','var')
    tformfname='40xto10x.mat';
end

load(tformfname,'tform40xto10x');
reg_folder_name='checkregistration';
mkdir(reg_folder_name)
uniqpos=sort_nat(unique(pos));
[~,posidx]=ismember(pos,uniqpos);
%
is_jpg=0;
f=dir([folders{1},'/RGB/*',fname,'*.tif']);
if isempty(f)
    f=dir([folders{1},'/RGB/*',fname,'*.jpg']);
    is_jpg=1;
end
f={f.name};

%check if the 10x folder is present
for i=1:numel(uniqpos)
    %get image size
    %%
    subfolders=find(posidx==i);
    if isfolder('10x')
        %use the 10x image size as the template
    img10xfilename=['10x/',uniqpos{i},'.tif'];
    img10xinfo=imfinfo(img10xfilename);
    im=repmat({zeros(img10xinfo(1).Height,img10xinfo(1).Width,3,'uint8')},numel(f),1);
    else
        T=zeros(numel(subfolders),2);
        for n=1:numel(subfolders)
        %calculate the expected 10x image size using the tforms.
        T(n,:)=tform40xto10x{subfolders(n)}.T(3,1:2);
        end
        max_T=max(T,[],1);%max translations, always positive
        uniq_x=numel(unique(floor(T(:,1)/1000))); % count number of tiles on x and y
        uniq_y=numel(unique(floor(T(:,2)/1000)));
        T_step=max_T./[uniq_x-1,uniq_y-1];%mean per tile translation
        im=repmat({zeros(ceil(T_step(2)*uniq_y+1),ceil(T_step(1)*uniq_x+1),3,'uint8')},numel(f),1);% make this tile num +1 so to accomodate the edge of the last image.
    end
    %%
    for n=1:numel(subfolders)
        if is_jpg>0
            imgfiles=dir([folders{subfolders(n)},'/RGB/*',fname,'*.jpg']);
        else
            imgfiles=dir([folders{subfolders(n)},'/RGB/*',fname,'*.tif']);
        end
        imgfiles={imgfiles.name};
        for m=1:numel(imgfiles)
            im1=imread([folders{subfolders(n)},'/RGB/',imgfiles{m}]);
            im1=imwarp(im1,tform40xto10x{subfolders(n)},'OutputView',imref2d([size(im{m},1),size(im{m},2)]));
            im1(1:10,1:10,:)=150;%add a border so that it's easy to count
            im{m}=max(im{m},im1);
        end
    end
    
    %%
    for m=1:numel(im)
        im{m}=im{m}*scaling;
        imwrite(imresize(uint8(im{m}),0.2),[reg_folder_name,'/',uniqpos{i},f{m}(1:end-4),'.tif']);
    end
end

    fprintf('Stitched images in folder: checkregistration. Visually check to make sure they are registered correctly.\n')

end