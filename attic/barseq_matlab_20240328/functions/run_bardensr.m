
function cmdout=run_bardensr(use_predefined_thresh)
    %This needs to be changed on new systems
    if ~exist('use_predefined_thresh','var')
        use_predefined_thresh=0;
    end



    py_root_n2v = fileparts("C:\barseq_envs\bardensr\python.exe");
    ENV = getenv('PATH');
    oldpath=ENV;
    ENV = strsplit(ENV, ';');
    items_to_add_to_path = {
        char(fullfile(py_root_n2v, 'Library', 'mingw64', 'bin'))
        char(fullfile(py_root_n2v, 'Library', 'usr', 'bin'))
        char(fullfile(py_root_n2v, 'Library', 'bin'))
        char(fullfile(py_root_n2v, 'Scripts'))
        char(py_root_n2v)
        };
    ENV = [items_to_add_to_path(:); ENV(:)];
    ENV = unique(ENV, 'stable');
    ENV = strjoin(ENV, ';');
    
    fprintf('Starting bardensr in python ...')
    cd ..
    setenv('PATH', ENV);
    if use_predefined_thresh
        [status,cmdout]=system('python bardensrbasecall_predefinedthresh.py','-echo');
    else
        [status,cmdout]=system('python bardensrbasecall.py','-echo');
    end
    setenv('PATH',oldpath);
    cd processed
    if status==0
        fprintf('bardensr finished successfully')
    else
        warning('bardensr has a warning. Please check cmdout.')
    end

end