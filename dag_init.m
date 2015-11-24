function [net] = dag_init(varargin)

    addpath ../matconvnet/matlab

    net = dagnn.DagNN();
    
    % Scene Recognition Part (AlexNet)
    conv1 = dagnn.Conv('size', [5 5 3 96], 'stride', [2 2]);
    net.addLayer('conv1', conv1, {'x1'}, {'x2'}, {});
    norm1 = dagnn.LRN();
    net.addLayer('norm1', norm1, {'x2'}, {'x3'}, {});
    mp1 = dagnn.Pooling('poolSize', [3 3], 'stride', [2 2]);
    net.addLayer('mp1', mp1, {'x3'}, {'x4'}, {});
    
    conv2 = dagnn.Conv('size', [5 5 48 256], 'pad', [2 2 2 2]);
    net.addLayer('conv2', conv2, {'x4'}, {'x5'}, {});
    norm2 = dagnn.LRN();
    net.addLayer('norm2', norm2, {'x5'}, {'x6'}, {});
    mp2 = dagnn.Pooling('poolSize', [3 3], 'stride', [2 2]);
    net.addLayer('mp2', mp2, {'x6'}, {'x7'}, {});
    
    conv3 = dagnn.Conv('size', [3 3 256 384], 'pad', [1 1 1 1]);
    net.addLayer('conv3', conv3, {'x7'}, {'x8'}, {});
    
    conv4 = dagnn.Conv('size', [3 3 192 384], 'pad', [1 1 1 1]);
    net.addLayer('conv4', conv4, {'x8'}, {'x9'}, {});
    
    conv5 = dagnn.Conv('size', [3 3 192 256], 'pad', [1 1 1 1]);
    net.addLayer('conv5', conv5, {'x9'}, {'x10'}, {});
    
    mp3 = dagnn.Pooling('poolSize', [3 3], 'stride', [2 2]);
    net.addLayer('mp3', mp3, {'x10'}, {'x11'}, {});
    
    conv6 = dagnn.Conv('size', [6 6 256 4096]);
    net.addLayer('conv6', conv6, {'x11'}, {'x12'}, {});
    dropout1 = dagnn.DropOut();
    net.addLayer('dropout1', dropout1, {'x12'}, {'x13'}, {});
    
    conv7 = dagnn.Conv('size', [1 1 4096 4096]);
    net.addLayer('conv7', conv7, {'x13'}, {'x14'}, {});
    dropout2 = dagnn.DropOut();
    net.addLayer('dropout2', dropout2, {'x14'}, {'x15'}, {});
    
    conv8 = dagnn.Conv('size', [1 1 4096 1000]);
    net.addLayer('conv8', conv8, {'x15'}, {'x16'}, {});
    
    % Object Recognition Part (TODO: should be parallel)
end

