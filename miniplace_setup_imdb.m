function imdb = miniplace_setup_imdb()

% Load miniCategoryIdx.mat & initialize imdb
%
dataDir = fullfile(fileparts(mfilename('fullpath')), '..', 'data') ;

load(fullfile(fileparts(mfilename('fullpath')),'miniCategoryIdx.mat'));
trainIdx = load(fullfile(fileparts(mfilename('fullpath')),'trainIdx.mat'));
valIdx = load(fullfile(fileparts(mfilename('fullpath')),'trainIdx.mat'));

imdb.classes.name = {miniCategoryIdx{:,1}} ;
imdb.classes.category = [miniCategoryIdx{:,2}] ;
imdb.imageDir = fullfile(dataDir, 'images') ;

% -------------------------------------------------------------------------
%                                                         Train images
% -------------------------------------------------------------------------
names = trainIdx.filename' ;
labels = trainIdx.category'+1 ; % matlab index

numTrain = numel(names);

imdb.images.id = 1:numTrain ;
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels ;

% -------------------------------------------------------------------------
%                                                         Validation images
% -------------------------------------------------------------------------
names = valIdx.filename' ;
labels = valIdx.category' +1 ; % matlab index

numVal = numel(names);

imdb.images.id = horzcat(imdb.images.id, (1:numVal) + numTrain - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numVal)) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;

% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------

ims = dir(fullfile(dataDir, 'images', 'test', '*.jpg')) ;
names = sort({ims.name}) ;
labels = zeros(1, numel(names)) ;

names = strcat(['test' filesep], names) ;
numTest = numel(names);

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + numTrain + numVal - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 3*ones(1,numTest)) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;