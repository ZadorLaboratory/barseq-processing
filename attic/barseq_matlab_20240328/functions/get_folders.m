

function [folders,pos,xxx,yyy]=get_folders()
    %get all folder names and corresponding slice names
    folders=dir('MAX*');
    folders=sort_nat({folders.name});
    pos={};
    xxx=[];
    yyy=[];
    for n=1:numel(folders)
        i1=regexp(folders{n},'_');
        pos{n}=folders{n}(i1(1)+1:i1(2)-1);
        xxx(n)=str2double(folders{n}(i1(2)+1:i1(3)-1));
        yyy(n)=str2double(folders{n}(i1(3)+1:end));
    end
    xxx=xxx+1; %xxx and yyy +1
    yyy=yyy+1; %xxx and yyy +1
    
end
