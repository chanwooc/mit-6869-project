function Y = vl_euclideanloss(X, c, dzdy)

  assert(numel(X) == numel(c));
  d = size(X);

  % assert(all(d == size(c)));
  X = reshape(X, size(c, 1), size(c, 2));

  if nargin == 2 || (nargin == 3 && isempty(dzdy))
%     load('tester.mat');
    Y = sum(subsref((X - c) .^ 2, substruct('()', {':'})));
%     if ~exist('Ys')
%       Ys = [];
%     end
%     Ys = [Ys; Y];
%     save('tester.mat', 'Ys');
  elseif nargin == 3 && ~isempty(dzdy)
    assert(numel(dzdy) == 1);
    Y = dzdy * (X - c);
  end

end
