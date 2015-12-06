function imdb = object_setup_imdb()

  addpath ../development_kit/util
  
%   imdb.imageDir = fullfile('..', 'data', 'images') ;
  imdb.imageDir = fullfile('.');

  %% ----------------------------------------------------------------------
  %                                                       Train images
  % -----------------------------------------------------------------------

  TRAIN_SIZE = 3502;
  names = [];
  labels = [];
  
  trainDir = '../data/objects/train/';
  trainData = getAllFiles(trainDir)';
  assert (size(trainData, 2) == TRAIN_SIZE);
  
  for t = trainData
    name = t{1,1};
    rec = VOCreadxml(name);
    imFilename = strrep(name, 'xml', 'jpg');
    imFilename = strrep(imFilename, 'objects', 'images');
    if isfield(rec, 'objects')
      names = [names cellstr(imFilename)];
      labels = [labels getLabel(rec.objects.class)];
    end
  end
  
  numVal = numel(names);
  imdb.images.id = 1:numVal;
  imdb.images.name = names;
  imdb.images.set = ones(1, numel(names));
  imdb.images.label = labels;

  %% ----------------------------------------------------------------------
  %                                                       Validation images
  % -----------------------------------------------------------------------

  VAL_SIZE = 371;
  names = [];
  labels = [];
  
  trainDir = '../data/objects/val/';
  trainData = getAllFiles(trainDir)';
  assert (size(trainData, 2) == VAL_SIZE);
  
  for t = trainData
    name = t{1,1};
    rec = VOCreadxml(name);
    imFilename = strrep(name, 'xml', 'jpg');
    imFilename = strrep(imFilename, 'objects', 'images');
    if isfield(rec, 'objects')
      names = [names cellstr(imFilename)];
      labels = [labels getLabel(rec.objects.class)];
    end
  end
  
  trainNumVal = numVal;
  numVal = numel(names);
  imdb.images.id = horzcat(imdb.images.id, (1:numVal) + trainNumVal - 1) ;
  imdb.images.name = horzcat(imdb.images.name, names) ;
  imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numVal)) ;
  imdb.images.label = horzcat(imdb.images.label, labels) ;

end

function label = getLabel(varargin)
  LABEL_SIZE = 175;
  label = zeros(LABEL_SIZE, 1);
  for var = varargin
    label(str2double(var{1,1}) + 1) = 100; % matlab index
  end
end

function fileList = getAllFiles(dirName)

  dirData = dir(dirName);
  dirIndex = [dirData.isdir];
  fileList = {dirData(~dirIndex).name}';
  if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirName,x),...
                       fileList,'UniformOutput',false);
  end
  subDirs = {dirData(dirIndex).name};
  validIndex = ~ismember(subDirs,{'.','..'});

  for iDir = find(validIndex)
    nextDir = fullfile(dirName,subDirs{iDir});
    fileList = [fileList; getAllFiles(nextDir)];
  end

end