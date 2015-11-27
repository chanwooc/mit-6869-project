function [net] = object_annot_init()

  % Bring configuration from simplenn
  netSimple = miniplace_init();
  net = dagnn.DagNN.fromSimpleNN(netSimple, 'canonicalNames', true);
  
%   rec = VOCreadxml('../data/objects/train/a/airport_terminal/00000001.xml');

end
