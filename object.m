function object(varargin)
  run(fullfile(fileparts(mfilename('fullpath')), ...
    '..','matconvnet', 'matlab', 'vl_setupnn.m'));

  NUM_AUGMENTS = 1;

  opts.dataDir = fullfile(fileparts(mfilename('fullpath')),'data');
  opts.batchNormalization = false;
  opts.weightInitMethod = 'gaussian';
%   opts.weightInitMethod = 'xavierimproved';
  [opts, varargin] = vl_argparse(opts, varargin);

  opts.expDir = fullfile('data', 'object');
  [opts, varargin] = vl_argparse(opts, varargin);

  opts.numFetchThreads = 12; %12
  opts.lite = false;
  opts.imdbPath = fullfile(opts.expDir, 'object-imdb.mat');
  opts.train.batchSize = 64;%256
  opts.train.numSubBatches = 1;
  opts.train.continue = true;
  opts.train.gpus = [];
  opts.train.prefetch = true;
  opts.train.sync = false;
  opts.train.cudnn = true;
  opts.train.expDir = opts.expDir;
  if ~opts.batchNormalization
    opts.train.learningRate = logspace(-2, -4, 60);
  else
    opts.train.learningRate = logspace(-1, -4, 20);
  end
  [opts, varargin] = vl_argparse(opts, varargin);

  opts.train.numEpochs = numel(opts.train.learningRate);
  opts = vl_argparse(opts, varargin);

  % -----------------------------------------------------------------------
  %                                                 Database initialization
  % -----------------------------------------------------------------------

  if exist(opts.imdbPath)
    imdb = load(opts.imdbPath);
  else
    imdb = object_setup_imdb();
    mkdir(opts.expDir);
    save(opts.imdbPath, '-struct', 'imdb');
  end

  % -----------------------------------------------------------------------
  %                                                  Network initialization
  % -----------------------------------------------------------------------

  net = object_init('batchNormalization', opts.batchNormalization, ...
                    'weightInitMethod', opts.weightInitMethod);
  bopts = net.normalization;
  bopts.numThreads = opts.numFetchThreads;

  % compute image statistics (mean, RGB covariances etc)
  imageStatsPath = fullfile(opts.expDir, 'object-imageStats.mat');
  if exist(imageStatsPath)
    load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance');
  else
    [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts);
    save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance');
  end

  % One can use the average RGB value, or use a different average for
  % each pixel
  %net.normalization.averageImage = averageImage;
  net.normalization.averageImage = rgbMean;
  
  % -----------------------------------------------------------------------
  %                                             Stochastic gradient descent
  % -----------------------------------------------------------------------

  [v,d] = eig(rgbCovariance);
  bopts.transformation = 'stretch';
  bopts.averageImage = rgbMean;
  bopts.rgbVariance = 0.1*sqrt(d)*v';

%   bopts.numAugments = NUM_AUGMENTS;
%   opts.train.numAugments = bopts.numAugments;

  useGpu = numel(opts.train.gpus) > 0;

  fn = getBatchSimpleNNWrapper(bopts);
  [net,info] = cnn_train_for_objects(net, imdb, fn, opts.train, ...
                                     'conserveMemory', true);
  
end

function fn = getBatchSimpleNNWrapper(opts)
  fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts);
end

function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
  images = strcat([imdb.imageDir filesep], imdb.images.name(batch));
  im = miniplace_get_batch(images, opts, 'prefetch', nargout == 0);
  if isfield(opts,'numAugments')
      labels = kron(imdb.images.label(batch), ones(1, opts.numAugments));
  else
      labels = imdb.images.label(:,batch);
  end
end

function fn = getBatchDagNNWrapper(opts, useGpu)
  fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu);
end

function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
  images = strcat([imdb.imageDir filesep], imdb.images.name(batch));
  im = miniplace_get_batch(images, opts, ...
                              'prefetch', nargout == 0);
  if nargout > 0
    if useGpu
      im = gpuArray(im);
    end
    inputs = {'input', im, 'label', imdb.images.label(batch)};
  end
end

function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
  train = find(imdb.images.set == 1);
  train = train(1: 10: end); %train = train(1: 101: end);
  bs = 256;
  fn = getBatchSimpleNNWrapper(opts);
  for t=1:bs:numel(train)
    batch_time = tic;
    batch = train(t:min(t+bs-1, numel(train)));
    fprintf('collecting image stats: batch starting with image %d ...', batch(1));
    temp = fn(imdb, batch);
    z = reshape(permute(temp,[3 1 2 4]),3,[]);
    n = size(z,2);
    avg{t} = mean(temp, 4);
    rgbm1{t} = sum(z,2)/n;
    rgbm2{t} = z*z'/n;
    batch_time = toc(batch_time);
    fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time);
  end
  averageImage = mean(cat(4,avg{:}),4);
  rgbm1 = mean(cat(2,rgbm1{:}),2);
  rgbm2 = mean(cat(3,rgbm2{:}),3);
  rgbMean = rgbm1;
  rgbCovariance = rgbm2 - rgbm1*rgbm1';
end
