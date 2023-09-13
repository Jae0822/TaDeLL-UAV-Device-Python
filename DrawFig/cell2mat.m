function [TaskList] = cell2mat(c)

% This is the function to turn cell (python list) into a mat with 0 and 1

TaskList = ones(1, numel(c));
for i = 1: numel(c)
    if isequal(c{i}, py.NoneType)
        TaskList(i) = 0;
    end
end
% return TaskList
