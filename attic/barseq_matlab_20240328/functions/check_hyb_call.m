
function check_hyb_call(slice,codebookhyb)
% manually check the image of a few FOVs
output_dir=fullfile('figs');
mkdir(output_dir);

if ~exist('codebookhyb','var')
    load('../codebookhyb.mat','codebookhyb');
end
load('genehyb.mat','lroihyb','idhyb');
uniqhybid=unique(idhyb{slice});
figure;
hold on;
for i=1:numel(uniqhybid)
    scatter(lroihyb{slice}(idhyb{slice}==i,1),lroihyb{slice}(idhyb{slice}==i,2),3,'o','filled');
end
pbaspect([1 1 1]);
legend(codebookhyb(:,1),'Location','eastoutside');
set(gca,'ydir','reverse');
exportgraphics(gcf,fullfile(output_dir,['Slice',num2str(slice),'.jpg']));
end