% tileCreator_6
% For Micro-Manager 2.0
% ver6b, allow uncertain number of tiles
% Woodbury scope (TE2000E) ~Focus -0.5 um/unit for 1 unit piezo (Ludl)
%
% tileCreater before ver6 only works for Micro-Manager 1.4 not 2.0
%
% Modified to MM2.0 based on tileCreater_5b
%   1. The fileName of structures were modified accordingly
%   2. Added function to modified single number to cell array for 
%      jsonencode for device posiiton (jsonencode MATLAB 2020a)
%
% Disciption: create .pos file for tiling
%
% !! Important note!!
%   Haven't tested multiple position for same image during tiling, not
%   sure
%   whether it works properly, will do it later (10252021)
%
% Note:
%   1. Tile creating using default XY/Z-stage in micro-manager
%   2. Only include focus in the first tile
%   3. If just for change posiiton name, set col & row as 1 
%
% Last update: 11232022 LY
%
% Add var if the variable doesnt exist
% Clear up the variable if its a table

clc

% (Mannual) Paste excel input to excelInput cell array
if ~exist('excelInput') || ~iscell(excelInput) || isempty(excelInput)
    excelInput = {};
    warning('Enter excelInput');
    openvar('excelInput');
    return
end

%% 0. Setting

% Camera pixel number in x & y
pixelNum = [1200 1200];

initValIs0 = true;

delimiter = '_';

% Name of the focus device (microscope stage z, not piezo/default Z)
focusDev = 'Focus';

% Stage with reverse axis
% (Currently only support xyStage)
reverseXY = [false,false];
reverseXY = reverseXY.*(-1);
reverseXY(reverseXY == 0) = 1;

% Output .pos file name append (string)
posFileNameAppend = '_tile';

% Get Directory and file name
[posFileName,posFileDirectory] = uigetfile('*.pos');

%% Import .pos file

% Open .pos file as structure
posIn = fopen(strcat(posFileDirectory,posFileName));
% mm2.0 require preserve empty space, i.e. "Micro-Manager Property Map"
posIn = textscan(posIn,'%s','Whitespace','\b');
posIn = posIn{1};
% Convert to structure
posIn = jsondecode(horzcat(posIn{:}));

pos = posIn.map.StagePositions.array;

%% Sample labels
% Note: 1. allow multipe position for single tiling grid
%       2. function for generate main label can be changed
%       3. The reason use excel because its faster and easier to check and
% modified/colorcode experiments

% Import position info from excel, convert to table
if iscell(excelInput)
    varName = excelInput(1,2:end);
    rowName = excelInput(2:end,1);
    excelInput = cell2table(excelInput(2:end,2:end),...
        'VariableNames',varName,'RowNames',rowName);
end

% Unique main labels of the samples
mainLabel = {};
for i = 1:size(excelInput,1)
    mainLabel{i,1} = getMainLabel(excelInput(i,:));
end

[mainLabel,~,ic] = unique(mainLabel);

%% Get tiling position

posOut = {};
i = 1;
while i<= numel(mainLabel)
    % Rows belongs the current tiling
    row = ic == i;
    row = find(row);
    
    % Tile/XYstage position (from excelInput) -----------------------------
    % Note: only use the first one to get tiling info
    % (for allowing multiple inputs for same position)
    iInput = excelInput(row(1),:);
    
    nRow = iInput.GRID_ROW; nCol = iInput.GRID_COL;
    overlapPercentage = iInput.OverlapPercentage;
    pixelSize = iInput.PixelSize;
    
    % Tile number and its position in the grid
    tilePos = getTilePosition(nRow,nCol);
    
    % Get xy-position normalized to center
    tileXY = getStagePos(tilePos,pixelNum,overlapPercentage,pixelSize);
    
    % Reverse axis
    tileXY = cellfun(@(X) X.*(reverseXY.*(-1)),tileXY,'UniformOutput',false);
    
    % Get position structure (from .pos) ==================================
    % Pos name in .pos file with this main label
    iLabel = excelInput.Properties.RowNames(row);
    
    % Get rows in position list
    posLabel = cellfun(@(X) X.scalar,{pos.Label},'UniformOutput',false);
    iPos = cellfun(@(X) strcmp(posLabel,X),iLabel,'UniformOutput',false);
    iPos = vertcat(iPos{:});
    iPos = any(iPos,1);
    iPos = pos(iPos,:);
    
    if isempty(iPos)
        warning(['Position .pos input with no label: ',mainLabel{i}]);
        continue
    end
    
    % Devices information: xy, z & focus ----------------------------------
    
    % Default XY/Z stages, for tiling
    dfXYdevice = iPos(1).DefaultXYStage.scalar;
    dfZdevice = iPos(1).DefaultZStage.scalar;
    
    % Devices
    dev = cellfun(@(X) X.array,{iPos.DevicePositions},'Uniformoutput',false);
    
    % Positon XYZ in .pos: row-xyz, col, position number
    posXYZ = [];
    for j = 1:numel(dev)
        % selectDevide(dev,devName)
        idev = selectDevide(dev{j},dfXYdevice);
        posXYZ(j,1:2) = idev.Position_um.array;
        
        idev = selectDevide(dev{j},dfZdevice);
        posXYZ(j,3) = idev.Position_um.array;
    end
    
    % Pick the first struct as sample for repmat
    dev = dev{1};
    
    % XY/Z/Focus devices
    devXY = selectDevide(dev,dfXYdevice);
    devZ = selectDevide(dev,dfZdevice);
    devFocus = selectDevide(dev,focusDev);
        
    % Tiling devices info: XYstage ----------------------------------------
    
    % xy center of the position
    xyCenter = mean(posXYZ(:,1:2),1);
    % xy-position for individual tiles
    tileXY = cellfun(@(X) X + xyCenter,tileXY,'UniformOutput',false);
    
    if iInput.trimTF
        minLim = min(posXYZ(:,1:2));
        maxLim = max(posXYZ(:,1:2));
        
        TF = cellfun(@(X) all(X > minLim) & all(X < maxLim),tileXY);
        % One tile out for edge effect
        TF = imdilate(TF,true(3));   
        
        sz = [max(sum(TF,1)),max(sum(TF,2))];
        tileXY = tileXY(TF); tileXY = reshape(tileXY,sz);
        
        tilePos = getTilePosition(sz(1),sz(2));
    end 
              
    % Write devices
    devXY = repmat({devXY},size(tileXY));
    for j = 1:numel(devXY)
        devXY{j}.Position_um.array = tileXY{j};
    end
    
    % Tiling devices info: Zstage -----------------------------------------
    % Regression if there is >= 3 position
    if size(posXYZ,1) >= 3
        fitobject = fit(posXYZ(:,1:2),posXYZ(:,3),'poly11');
        tileZ = cellfun(@(X) feval(fitobject,X),tileXY);
    else
        tileZ = mean(posXYZ(:,3));
        tileZ = repmat(tileZ,size(tileXY));
    end
    
    % Write devices
    devZ = repmat({devZ},size(tileXY));
    for j = 1:numel(devZ)
        devZ{j}.Position_um.array = tileZ(j);
    end
           
    % Comebine XY & Z-devices ---------------------------------------------
    dev = cellfun(@(X,Y) [X,Y],devZ,devXY,'UniformOutput',false);
    
    % Get col & row in tiling name ----------------------------------------
    % 5b, change to use col & row; I, tile number
    [row,col,I] = find(tilePos);   
   
    % Reverse axis & set initial col and row to 0
    if reverseXY(1)~= 0
        col = col.*(-1); 
    end  
    col = col - min(col);
    
    % Y-axis is the reverse of row direction
    if reverseXY(2) ~= 0
        row = row.*(-1);         
    end        
    row = row - min(row);
    
    % Set inital row/col as 1
    if ~initValIs0
        row = row +1;
        col = col +1;
    end
    
    colRow = arrayfun(@(X,Y)[num2str(X),delimiter,num2str(Y)],...
        col,row,'Uniformoutput',false);
        
    % Sort using tile number
    dev = reshape(dev,[],1);
    
    [~,I] = sort(I,'ascend');
    dev = dev(I);
    colRow = colRow(I);
    
    % Add focus to 1st tile
    dev{1} = [devFocus,dev{1}];
    
    % Pos output ==========================================================
    iPosOut = repmat(iPos(1),size(dev));
    
    % Change device and label
    for j = 1:size(iPosOut,1)
        iPosOut(j).DevicePositions.array = dev{j};
        iPosOut(j).Label.scalar = [mainLabel{i},delimiter,colRow{j}];
    end
    
    posOut{i,1} = iPosOut;     
    i = i + 1;
end

posOut = vertcat(posOut{:});

% Output .pos file --------------------------------------------------------
% Only change the POSITIONS, keep the rest
pos = posIn;
pos.map.StagePositions.array = posOut;

%% Correct format before jsonencode
pos = correctFormat(pos);
% Convert structure to txt
pos = jsonencode(pos);

% Output name
posOutName = erase(posFileName,'.pos');
posOutName = [posOutName,posFileNameAppend];

% Writing .pos file
% cannot directly write to .pos file using writecell, so copy txt file to
% pos file, then delete the txt file
cd(posFileDirectory);
writecell({pos},posOutName,'QuoteStrings',false);
copyfile([posOutName,'.txt'],[posOutName,'.pos']);
delete([posOutName,'.txt']);

%% Function:    getMainLabel
% Discription:  Generate main laybe for the postion
function mainLabel = getMainLabel(tblIn)
% Input:    tblIn, a row of table, with MouseID, SlideNumber, SectionPos,
% PrependTxt, PrependNumber, Location
% Output:   mainLabel, str, label of the sample

MouseID = tblIn.MouseID;

% Slide number & seciton position per slide
Slide = ['Slide',num2str(tblIn.SlideNumber)];
Section = ['Section',tblIn.SectionPos{:}];

% Prepend (i.e. Seq01, Ab02)
Prepend = [tblIn.PrependTxt{:},...
    num2str(tblIn.PrependNumber,'%02d')];

% Image location in the sample
imageLoc = tblIn.Location{:};

% Delemiter is '_'
mainLabel = strjoin([MouseID,Slide,Section,Prepend,imageLoc],'_');

end

%% Function:    getTilePosition
% Discription:  get tile position using number of rows and columns
% (sneak-right-up)
function tilePos = getTilePosition(nRow,nColumn)
% Input:    nRow/nColumn, num of row and column for the grid
% Output:   tilePos, mat, updated tilePos

nGrid = nRow*nColumn; % Number of grid

% Construct current tile positions
tilePos = 1:nGrid;
tilePos = reshape(tilePos,nColumn,nRow)';
if size(tilePos,1) >= 2
    tilePos(1:2:end,:) = fliplr(tilePos(1:2:end,:));
end
tilePos = flipud(tilePos);
end

%% Function:    getStagePos
% Discription:  generate xy stage position according to tilePos
function stagePos = getStagePos(tilePos,pixelNum,overlapPercentage,pixelSize)
% Input:    tilePos, mat, with tile number in tiling grid
%           pixelNum, vector, camera pixel number
%           overlapPrecentage, num, precentage of overlap during tiling
%           pixelSize, num, how many micron per pixel, for calculating
%           stage movement
% Output:   stagePos, cell, same size as tilePos, with xy position per cell
%           normalzied to center

% In case the precentage input >1
if overlapPercentage > 1
    overlapPercentage = overlapPercentage.*0.01;
end

stagePos = cell(size(tilePos));
for i = 1:numel(tilePos)
    I = tilePos == i;
    [y,x] = find(I);
    xy = [x y];
    stagePos{I} = xy - (xy-1).*overlapPercentage;
end

% Pixel position
stagePos = cellfun(@(X) (X-1).*pixelNum,stagePos,'UniformOutput',false);

% Convert pixel position to micron position
stagePos = cellfun(@(X) X.*pixelSize,stagePos,'UniformOutput',false);

% Normalized to center
center = vertcat(stagePos{:});
center = mean(center,1);
stagePos = cellfun(@(X) X-center,stagePos,'UniformOutput',false);

end

%% Function:    selectDevice
% Discription:  select device using device name
function dev = selectDevide(dev,devName)
% Input & Output: dev, structure array, with device name and info of axes
%               devName, str, device name

devIn = cellfun(@(X) X.scalar,{dev.Device},'Uniformoutput',false);

TF = strcmp(devIn,devName);
dev = dev(TF);

end

%% Function:    correctFormat
% Discription:  correct format for MATLAB jsonencode 2020a 
function pos = correctFormat(pos)
% To change single number into cell array under stagePosition.array
% Input & output:   pos, struct

posOut = pos.map.StagePositions.array;

for i = 1:numel(posOut)
    iPos = posOut(i).DevicePositions.array;
    
    for j = 1:numel(iPos)
        % Only set single number into cell array, multiple number array
        % cannot be set into cell array
        jPos = iPos(j).Position_um.array;
        if numel(jPos) == 1
            iPos(j).Position_um.array = {jPos};
        end
    end
    
    posOut(i).DevicePositions.array = iPos;
end

pos.map.StagePositions.array = posOut;

end
