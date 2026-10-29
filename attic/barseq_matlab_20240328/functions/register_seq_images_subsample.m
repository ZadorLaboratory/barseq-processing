
function register_seq_images_subsample(fname,chprofile20x,ball_radius,chshift20x,rgb_intensity,local,subsample_rate)
if ~exist('subsample_rate','var')
    subsample_rate=1;
end
if ~exist('local','var')
    local=1;
end
folders=get_folders();
%worker pools

%   p=gcp('nocreate');
%  fprintf('Starting image registration on %u workers...\n',p.NumWorkers)
% tic
parfor i=1:length(folders)
    %for i=1
    %    for i=13
    cd(folders{i});
    if isfolder('original')
        cd original
        try
            movefile([fname,'*.tif'],'../')
        catch ME
            %rethrow(ME)
        end
        cd ..
    end
    if local==1
        [~,~,warnmsg]=mmseqalignmentsinglethread_local_subsample(fname,chprofile20x,ball_radius,chshift20x,subsample_rate);
    else
        [~,~,warnmsg]=mmseqalignmentsinglethread(fname,chprofile20x,ball_radius,chshift20x);
    end

    if ~isempty(warnmsg)
        warning('%s: %s\n',folders{i},warnmsg);
    end
    cd aligned
    rgbout1(['alignedfixed',fname],rgb_intensity);
    mkdir('../RGB');
    movefile('RGB*.tif','../RGB/');
    cd ../..
end
regtime=toc;
fprintf('Registration finished, total elapsed time is %u hours %u minutes %u seconds',round(regtime/3600),round(rem(regtime,3600)/60),round(rem(regtime,60)))
end