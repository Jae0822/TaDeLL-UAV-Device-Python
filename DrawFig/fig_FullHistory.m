% This is the matlab code to draw full history figure
% The original figure is in fig_FullHistory.py


%% Prepare Data
pysys = py.sys.path;
pysys.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
numpy = py.importlib.import_module('numpy');
pickle = py.importlib.import_module('pickle');

fh = py.open('output/050923-2009-6_devices-tadell_model/output.pkl', 'rb');
P = pickle.load(fh);
% model, env, env_random, env_force, param, avg, logging_timeline = pickle.load(fh); 
fh.close();


P_cell = cell(P);
param = P_cell{5};
logging_timeline = P_cell{7};


%% Painting Flow
% Create a tiledlayout
figure
t = tiledlayout('flow');
t.TileSpacing = 'tight';
% Plot in tiles
ep = double(param{'episodes'});
for x  =  1:double(param{'num_Devices'})
% for x  =  1:6
    nexttile, stairs(logging_timeline{x}{ep}{'KeyTime'}, logging_timeline{x}{ep}{'KeyRewards'}, 'LineWidth',2,'Marker','d','MarkerFaceColor','c')
    % xlabel('dsdfs')
    title('device ' +  string(x))
    % ylabel('dsdsds')
    % Task change line in vertical
    c = cell(logging_timeline{x}{ep}{'TaskList'});
    xline(find(cell2mat(c) == 1), '-.b')
    % The average reward in horizontal
    yline(logging_timeline{x}{ep}{'avg_reward'}, 'Color', "#D95319", 'LineWidth', 1.5, 'LineStyle', '-.', 'Label', 'Averaged Values', 'LabelVerticalAlignment','bottom')
    % Look better
    xlim([0, numel(cell(logging_timeline{x}{ep}{'TaskList'}))])
    ylim([-inf, inf])
    % legend( 'Averaged Values', 'Location','best')
    % ylabel('device ' +  string(x))
end
% Specify common title, X and Y labels
% title(t, 'Common title')
xlabel(t, 'Number of Time Slots', 'FontSize', 16)
ylabel(t, 'Averaged Reward','FontSize', 16)



%% Painting Vertical
% Create a tiledlayout
figure
t = tiledlayout('vertical');
% Plot in tiles
ep = double(param{'episodes'});
% for x  =  1:double(param{'num_Devices'})
for x  =  1:5
    nexttile, 
    stairs(logging_timeline{x}{ep}{'KeyTime'}, logging_timeline{x}{ep}{'KeyRewards'}, 'LineWidth',2,'Marker','d','MarkerFaceColor','c');
    % xlabel('dsdfs')
    % Task change line in vertical
    c = cell(logging_timeline{x}{ep}{'TaskList'});
    xline(find(cell2mat(c) == 1), '-.b')
    % The average reward in horizontal
    yline(logging_timeline{x}{ep}{'avg_reward'}, 'Color', "#D95319", 'LineWidth', 1.5, 'LineStyle', '-', 'Label', 'Averaged Values', 'LabelVerticalAlignment','bottom')
    % Look better
    xlim([0, numel(cell(logging_timeline{x}{ep}{'TaskList'}))])
    ylim([-inf, inf])
    ylabel('device ' +  string(x))
end
% Specify common title, X and Y labels
% title(t, 'Common title')
xlabel(t, 'Number of Time Slots','FontSize', 16)
ylabel(t, 'Averaged Reward', 'FontSize', 16)
t.TileSpacing = 'tight';

%% Painting

% fig = figure;
% hold on
% 
% ep = double(param{'episodes'});
% for x = 1:double(param{'num_Devices'})
%     axx = subplot(double(param{'num_Devices'}), 1, x);
%     stairs(logging_timeline{x}{ep}{'KeyTime'}, logging_timeline{x}{ep}{'KeyRewards'}, 'LineWidth',2,'Marker','d','MarkerFaceColor','c')
%     % Task change line in vertical
%     c = cell(logging_timeline{x}{ep}{'TaskList'});
%     xline(find(cell2mat(c) == 1), '-.b')
%     % The average reward in horizontal
%     yline(logging_timeline{x}{ep}{'avg_reward'}, 'r', 'LineWidth', 1.5)
%     % Look better
%     xlim([0, numel(cell(logging_timeline{x}{ep}{'TaskList'}))])
%     ylim([-inf, inf])
%     ylabel('device ' +  string(x))
%     % axx.FontSize = 10;
% end






d = 1;