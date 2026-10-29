function register_hyb_images_multi(ch,radius,nuclearch,reg_cycle,chradius,chprofilehyb,chshifthyb, rgbintensity)
    %register hyb to first seq cycle, copy nuclear images
    fname='n2vhyb';
    
    numcores=feature('numcores');

    p=gcp('nocreate');
    if isempty(p)
        p= parpool("local",round(numcores*1.5));
    elseif p.NumWorkers<round(numcores*1.5)
        p= parpool("local",round(numcores*1.5));
    end
    folders=get_folders();
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
        cd ..
    end

    fprintf(['Registering hyb to seq cycle ', num2str(reg_cycle),' on ', num2str(p.NumWorkers) ' workers...\n'])
     %Aixin added here
    cd(folders{1})
    hybfiles_n=numel(dir([fname,'*.tif']));
    %fprintf('%d\n',hybfiles_n)
    %fprintf('%d\n',numel(ch))
    
    if numel(ch)==1 && hybfiles_n>1
       ch = repmat(ch,1,hybfiles_n);
    end
    cd ..
    if numel(ch) ~= hybfiles_n
       error('Hyb cycle number is not consistent with hyb_reg_ch! ')
    end
       %
    parfor i=1:length(folders)
    %for i=39
        cd(folders{i})
        mmalignhybtoseq_local(fname,'geneseq',ch,chprofilehyb,radius,nuclearch,reg_cycle,chradius,chshifthyb);
        rgbout1('aligned',rgbintensity);
        movefile('aligned*.tif','aligned/');
        movefile('RGB*.tif','RGB/');
        cd ..
    
    end
    fprintf('Registration finished, copying files to make a hyb copy ...\n')
    % Make sequential rounds images
    for i=1:length(folders)
        copyfile([folders{i},'/aligned/alignedn2vhyb01.tif'],[folders{i},'/aligned/alignednuclear.tif']);
    end
    fprintf('All done!\n')
end