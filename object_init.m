function net = object_init(varargin)

  % General options
  opts.scale = 100 ;
  opts.weightDecay = 1 ;
  %opts.weightInitMethod = 'xavierimproved' ;
  opts.weightInitMethod = 'gaussian' ;
  opts.batchNormalization = false ;
  opts = vl_argparse(opts, varargin) ;

  % Define layers
  net.normalization.imageSize = [115, 115, 3] ;
  net = alexnet_object(net, opts); % NOTE: NO softmax
%   net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;
  net.layers{end+1} = struct('type', 'euclideanloss', 'name', 'loss');
                             
  net.normalization.border = 128 - net.normalization.imageSize(1:2) ;

  net.normalization.interpolation = 'bicubic' ;
  net.normalization.averageImage = [] ;
  net.normalization.keepAspect = true ;
end
 
function net = add_block(net, opts, id, h, w, in, out, stride, pad)
  info = vl_simplenn_display(net) ;
  fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
  if fc
    name = 'fc' ;
  else
    name = 'conv' ;
  end
  net.layers{end+1} = struct('type', 'conv', 'name', sprintf('%s%s', name, id), ...
                             'weights', {{init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')}}, ...
                             'stride', stride, ...
                             'pad', pad, ...
                             'learningRate', [1 2], ...
                             'weightDecay', [opts.weightDecay 0]) ;
  if opts.batchNormalization
    net.layers{end+1} = struct('type', 'bnorm', 'name', sprintf('bn%d',id), ...
                               'weights', {{ones(out, 1, 'single'), zeros(out, 1, 'single')}}, ...
                               'learningRate', [2 1], ...
                               'weightDecay', [0 0]) ;
  end
  net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%s',id)) ;
end

function weights = init_weight(opts, h, w, in, out, type)
  % See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
  % rectifiers: Surpassing human-level performance on imagenet
  % classification. CoRR, (arXiv:1502.01852v1), 2015.

  switch lower(opts.weightInitMethod)
    case 'gaussian'
      sc = 0.01/opts.scale ;
      weights = randn(h, w, in, out, type)*sc;
    case 'xavier'
      sc = sqrt(3/(h*w*in)) ;
      weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
    case 'xavierimproved'
      sc = sqrt(2/(h*w*out)) ;
      weights = randn(h, w, in, out, type)*sc ;
    otherwise
      error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
  end
end

function net = add_fc(net, name, size, stride, pad)
  net.layers{end+1} = struct('type', 'pool', ...
                             'name', sprintf('pool%s', name), ...
                             'method', 'max', ...
                             'pool', [size size], ...
                             'stride', stride, ...
                             'pad', pad) ;
end

function net = add_norm(net, opts, id)
  if ~opts.batchNormalization
    net.layers{end+1} = struct('type', 'normalize', ...
                               'name', sprintf('norm%s', id), ...
                               'param', [5 1 0.0001/5 0.75]) ;
  end
end

function net = add_dropout(net, opts, id)
  if ~opts.batchNormalization
    net.layers{end+1} = struct('type', 'dropout', ...
                               'name', sprintf('dropout%s', id), ...
                               'rate', 0.5) ;
  end
end

function net = alexnet_object(net, opts)

  LABEL_SIZE = 175;

  net.layers = {} ;

  net = add_block(net, opts, '1', 7, 7, 3, 64, 2, 0) ;
  net = add_norm(net, opts, '1') ;
  net = add_fc(net, '1', 3, 2, 0);

  net = add_block(net, opts, '2', 5, 5, 32, 128, 1, 2) ;
  net = add_norm(net, opts, '2') ;
  net = add_fc(net, '2', 3, 2, 0);

  net = add_block(net, opts, '3', 3, 3, 128, 192, 1, 1) ;
  net = add_block(net, opts, '4', 3, 3, 96, 128, 1, 1) ;
  net = add_fc(net, '3', 3, 2, 0);

  net = add_block(net, opts, '5', 6, 6, 128, 1024, 1, 0) ;
  net = add_dropout(net, opts, '5') ;

  net = add_block(net, opts, '6', 1, 1, 1024, 1024, 1, 0) ;
  net = add_dropout(net, opts, '6') ;

  net = add_block(net, opts, '7', 1, 1, 1024, LABEL_SIZE, 1, 0) ;
%   net.layers(end) = [] ;
  if opts.batchNormalization, net.layers(end) = [] ; end
end
