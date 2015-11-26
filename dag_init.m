function [net] = dag_init(varargin)

  % Bring configuration from simplenn
  % net = miniplace_init(varargin);    
  net = miniplace_init();
  netDag = dagnn.DagNN.fromSimpleNN(net);

end
