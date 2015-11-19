clear; clc;

% Load miniCategoryIdx.mat & initialize imdb
%
dataDir = fullfile(fileparts(mfilename('fullpath')), '..', 'data') ;
load(fullfile(fileparts(mfilename('fullpath')),'miniCategoryIdx.mat'));


imdb.classes.name = {miniCategoryIdx{:,1}} ;
imdb.classes.description = {miniCategoryIdx{:,2}} ;
imdb.imageDir = fullfile(dataDir, 'images') ;

% -------------------------------------------------------------------------
%                                                         Train images
% -------------------------------------------------------------------------
names = {} ;
labels = {} ;
for i = 1:size(miniCategoryIdx,1)
    ims = dir(fullfile(dataDir,'images','train',miniCategoryIdx{i,1}, '*.jpg'));
    names{end+1} = strcat([miniCategoryIdx{i,1},filesep], {ims.name});
    labels{end+1} = ones(1, numel(ims)) * i;
end

names = horzcat(names{:}) ;
labels = horzcat(labels{:}) ;

names = strcat(['train' filesep], names) ;

imdb.images.id = 1:numel(names) ;
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels ;

% -------------------------------------------------------------------------
%                                                         Validation images
% -------------------------------------------------------------------------

ims = dir(fullfile(dataDir, 'images', 'val', '*.jpg')) ;
names = sort({ims.name}) ;
labels = textread(valLabelsPath, '%d') ;

if numel(ims) ~= 50e3
  warning('Found %d instead of 50,000 validation images. Dropping validation set.', numel(ims))
  names = {} ;
  labels =[] ;
else
  if ~isempty(valBlacklistPath)
    black = textread(valBlacklistPath, '%d') ;
    fprintf('blacklisting %d validation images\n', numel(black)) ;
    keep = setdiff(1:numel(names), black) ;
    names = names(keep) ;
    labels = labels(keep) ;
  end
end

names = strcat(['val' filesep], names) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 1e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels') ;

% -------------------------------------------------------------------------
%                                                               Test images
% -------------------------------------------------------------------------

ims = dir(fullfile(opts.dataDir, 'images', 'test', '*.JPEG')) ;
names = sort({ims.name}) ;
labels = zeros(1, numel(names)) ;

if numel(labels) ~= 100e3
  warning('Found %d instead of 100,000 test images', numel(labels))
end

names = strcat(['test' filesep], names) ;

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 2e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 3*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels) ;