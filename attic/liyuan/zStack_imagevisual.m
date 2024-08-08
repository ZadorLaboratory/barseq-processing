% Require: Statistics and machine learning toolbox
% 02222024 LY

clc

% Flexible settings
directory = ['D:\barseq\ZD073124_floatingfixedOB\bcseq03']
workerNum = 25;

% Channels for QC
ch4QC = 1:5;

% Only can choose one item: "sampleInterval" or "selecedXY"
% Interval for downsampling
% sampleInterval = [10];
sampleInterval = [];

% For checking selected xy-position
% selecedXY = [];
selecedXY = [17];

% Task type
task = 'ImageVisualization';
%task = 'QCzstack';

%% Fixed settings ---------------------------------------------------------

% Ratio for scaling down the image to boost speed
scaleRatio = 0.1;

% Image format
imFormat = '.tif';

% Position of xy,z,c in file name
nameElement = [];
nameElement.xy = 1;
nameElement.z = 2;
nameElement.c = 3;

% Total channel number for QC
nCh = numel(ch4QC);

disp(['Checking diretory: ',directory]);

if ~isempty(sampleInterval) && ~isempty(selecedXY)
    error('Downsampling(sampleInterval) and selectXY cannot be done together. Please choos one of them.')
end

if ~isempty(selecedXY) && iscell(selecedXY)
    selecedXY = cellfun(@str2num,selecedXY);
end

% Task types: 1-image visualization; 2-QC zstack
if strcmp(task,'ImageVisualization')
    taskNum = 1;
    disp('Start checking individual image...');

elseif strcmp(task,'QCzstack')
    taskNum = 2;
    disp('Start z-stack QC...');

else
    error('Task name is not right.')
end

% Start MIJI for image visualization
if taskNum == 1
    if ~ispc
        error('Image visualization only supported in Windows system.');
    else
        startMIJ;
    end
end

try
    parpool(workerNum);
end

%% Code -------------------------------------------------------------------

% Get unique position
cd(directory);
fileName = dir(['*',imFormat]);
fileName = {fileName.name};
fileName = reshape(fileName,[],1);

% Only include selected channel
% (to boost speed by skipping reading image from unselected channels)
[~,~,ch] = getXYZC(fileName,nameElement);
TF = ismember(ch,ch4QC);
fileName = fileName(TF);

% Sort xy-c-z from low to high
[xy,z,c] = getXYZC(fileName,nameElement);
xycz = [xy,c,z];
[~,I] = sortrows(xycz,'ascend');
fileName = fileName(I);
xy = xy(I);

% (Optional) Downsample images for QC
if ~isempty(sampleInterval)    
    [C,~,ic] = unique(xy);
    % Start from the middle of the interval, because the first tile
    % generally is pretty empty
    TF = round(sampleInterval/2):sampleInterval:numel(C);
    TF = ismember(ic,TF);

    fileName = fileName(TF);    
end

% (Optional) Only work on selected positions
if ~isempty(selecedXY)
    TF = ismember(xy,selecedXY);
    fileName = fileName(TF);
end

[xy,~,~] = getXYZC(fileName,nameElement);
[~,~,ic] = unique(xy);

%% 
% Note: parfor is for reading image

vQC = [];
posNumber = [];
for i = 1:max(ic) 

    TF = ic == i;    
    iFileName = fileName(TF);

    iIm = getImage(iFileName,scaleRatio);

    if taskNum == 1
        MIJ.createImage(iIm);
        formatMIJimage(iFileName,nameElement);
        continue
    end

    % Evaluate z-position -------------------------------------------------
    % vQC, QC value. Use std to evlauate signal location, empty space
    % generally has low std)
    ivQC = single(iIm);
    ivQC = std(ivQC,[],1:2);
    ivQC = reshape(ivQC,[],nCh);

    % Use zscore instead of raw intensity, for each channel
    ivQC = zscore(ivQC,1);
    ivQC = median(ivQC,2);

    % Output --------------------------------------------------------------
    xy = getXYZC(iFileName,nameElement);

    vQC(i,:) = reshape(ivQC,1,[]);
    posNumber(i,1) = xy(1);
    disp(['Checked postion: xy ',num2str(xy(1))]);
end

if isempty(vQC)
    return
end

% Plot QC-value (3D plot) -------------------------------------------------
% Only run this for Widows system
if ispc
    plotQCvalue(vQC);
end

% QC using zscore = 0 -----------------------------------------------------
% Output in log file if run in Elzar
TF = vQC(:,1) > 0 | vQC(:,end) > 0;
if any(TF)
    warningList = posNumber(TF);
    disp('Signal has been detected close to the edge, please check warningList:');
    disp(warningList);
else
    warningList = [];
end

disp('displaying vQC matrix (row-sample number, col-zstack number):')
disp([posNumber vQC]);

%% Function:    startMIJ
% Descirption:  turn on MIJI for image visualization
function startMIJ
try
    TF = MIJ.version;

catch
    javaaddpath 'C:\Program Files\MATLAB\R2022b\java\mij.jar'
    javaaddpath 'C:\Program Files\MATLAB\R2022b\java\ij.jar'

    addpath 'C:\Users\Nikon\Downloads\fiji-win64\Fiji.app\scripts'
    Miji
end

end

%% Function:    formatMIJimage
% Description:  format the image window in MIJ
function formatMIJimage(fileName,nameElement)

[xy,z,c] = getXYZC(fileName,nameElement);

% Rename image window
MIJ.run('Rename...',strcat("title=xy",num2str(xy(1))));

% Change to hyperstack
nZ = max(z);
nCh = unique(c);
nCh = numel(nCh);
MIJ.run("Stack to Hyperstack...", ...
    strcat("order=xyzct channels=",num2str(nCh),...
    " slices=",num2str(nZ)," frames=1 display=Grayscale"));
end

%% Function:    getXYZC
% Description:  get xy/z/c info from file name, Marks Nikon scope
function [xy,z,c] = getXYZC(name,nameElement)
% Input:    name, cell,
%           nameElement, struct, with position for extracting xy/z/c info
% Output:   xy/z/c, vector, position/z-stack/channel number

if ~iscell(name)
    name = {name};
end

xyzc = cellfun(@(X) strsplit(X,{'xy','z','c','.'}),name,'Uniformoutput',false);
xyzc = vertcat(xyzc{:});

% Delete empty column
TF = cellfun(@isempty,xyzc);
TF = all(TF,1);
xyzc = xyzc(:,~TF);

% Extract element from file name
xy = xyzc(:,nameElement.xy);
z = xyzc(:,nameElement.z);
c = xyzc(:,nameElement.c);

% Change to matrix
xy = cellfun(@str2num,xy);
z = cellfun(@str2num,z);
c = cellfun(@str2num,c);
end

%% Function:    getImage
% Description:  get images using file names (with scaling)
function im = getImage(fileNames,scaleRatio)
% Input:    fileNames, cell
%           scaleRatio, num, scale factor
% Output:   im, mat, image stack

% Image size
nZ = numel(fileNames);
f = imfinfo(fileNames{1});
sz = [f.Height,f.Width,nZ];

% Image bits
imBits = f.BitDepth;
imBits = ['uint',num2str(imBits)];

im = zeros(sz,imBits);
parfor i = 1:size(im,3) % parfor
    im(:,:,i) = imread(fileNames{i});
end

% Scale down to speed up
if scaleRatio ~= 1
    im = imresize(im,scaleRatio);
end

end

%% Function:    plotQCvalue
% Description: make 3D line plots for QC value
function plotQCvalue(vQC)
% Input: vQC, mat, QC values. row: sample number; col: z-stack number

% x-axis: sample number
x = 1:size(vQC,1);
x = x';
x = repmat(x,1,size(vQC,2));

% y-axis: z-stack number
y = 1:size(vQC,2);
y = repmat(y,size(vQC,1),1);

% z-axis: QC value
z = vQC;

figure; plot3(x',y',z');
xlabel('x-axis: xy#'); ylabel('y-axis: z-stack#'); zlabel('z-axis: zscore');
g = gca; g.YDir = 'reverse';
view(-90,0);

end
