function cmdout=n2v_processing(fname)
    if ~exist('fname','var')
        fname='n2vprocessing.py';
    end

    %This path works for BARseq_envs
    py_root_n2v = fileparts("C:\barseq_envs\n2v\python.exe");
    ENV = getenv('PATH');
    oldpath=ENV;
    ENV = strsplit(ENV, ';');
    items_to_add_to_path = {
        char(fullfile(py_root_n2v, 'Library', 'mingw-w64', 'bin'))
        char(fullfile(py_root_n2v, 'Library', 'usr', 'bin'))
        char(fullfile(py_root_n2v, 'Library', 'bin'))
        char(fullfile(py_root_n2v, 'Scripts'))
        char(py_root_n2v)
        };
    ENV = [items_to_add_to_path(:); ENV(:)];
    ENV = unique(ENV, 'stable');
    ENV = strjoin(ENV, ';');
    
    fprintf('Starting n2v in python ...')
    
    setenv('PATH', ENV);
    [status,cmdout]=system(['python ',fname],'-echo');
    setenv('PATH',oldpath);
    if status==0
        fprintf('N2v finished successfully')
    else
        warning('N2v has a warning. Please check cmdout.')
    end

end