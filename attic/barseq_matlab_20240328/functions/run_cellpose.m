function cmdout=run_cellpose(script_name)
    if ~exist('script_name','var')
        script_name='Cellsegmentation-v065.py';
    end


%should work for BARseq_envs 
    py_root_n2v = fileparts("C:\BARseq_envs\cellpose\python.exe");
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
    
    fprintf('Starting cellpose in python ...')
    cd ..
    setenv('PATH', ENV);
    [status,cmdout]=system(['python ',script_name],'-echo');
    setenv('PATH',oldpath);
    cd processed
    if status==0
        fprintf('Cellpose finished successfully')
    else
        warning('Cellpose has a warning. Please check cmdout.')
    end
end