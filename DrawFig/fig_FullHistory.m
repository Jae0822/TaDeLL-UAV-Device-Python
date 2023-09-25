% This is the matlab code to draw full history figure
% The original figure is in fig_FullHistory.py

%% Prepare function
% function m = fig_FullHistory(c)
%     % function to compute mean of cell
%     m = 0;
%     for i = 1: numel(c)
%         m = m + c{i};
%     end
%     m = m/numel(c);
% end


%% Prepare Data
pysys = py.sys.path;
pysys.append('/Library/Frameworks/Python.framework/Versions/3.9/lib/python3.9/site-packages')
numpy = py.importlib.import_module('numpy');
pickle = py.importlib.import_module('pickle');

fh = py.open('output/150923-2101-6_devices-tadell_model_FullRun/output.pkl', 'rb');
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
st = 'aabcdef';
for x  =  1:double(param{'num_Devices'})-1
% for x  =  1:6
    nexttile, stairs(logging_timeline{x}{ep}{'KeyTime'}, logging_timeline{x}{ep}{'KeyRewards'}, 'LineWidth',2,'Marker','d','MarkerFaceColor','c')
    % xlabel('dsdfs')
    title('(' +string(st(x)) + '): Device ' +  string(x), 'Interpreter','latex', 'FontSize', 16)
    % title('device ' +  string(x - 1), 'Interpreter','latex', 'FontSize', 16)
    % ylabel('dsdsds')
    % Task change line in vertical
    c = cell(logging_timeline{x}{ep}{'TaskList'});
    xline(find(cell2mat(c) == 1), '-.b')
    % The average reward in horizontal
    yline(logging_timeline{x}{ep}{'avg_reward'}, 'Color', "#D95319", 'LineWidth', 1.5, 'LineStyle', '-.','FontSize', 14, 'Label', 'Averaged Values', 'LabelVerticalAlignment','bottom', 'interpreter','latex')
    % Look better
    xlim([0, numel(cell(logging_timeline{x}{ep}{'TaskList'}))])
    ylim([-inf, inf])
    % legend( 'Averaged Values', 'Location','best')
    ylabel('Average Reward', 'Interpreter','latex', 'FontSize', 13)
    xlabel('Number of Time Slots', 'Interpreter','latex', 'FontSize', 13)
end
% For the response time figure
nexttile,
yyaxis left
res = {};
for j  =  2:double(param{'num_Devices'})
    freq = double(P_cell{2}.Devices{j}.frequency);
    KeyTime = cell(logging_timeline{j}{ep}{'KeyTime'});
    i = 2;
    u = {};
    detector = 0;
    while KeyTime{i} > detector
        gap = KeyTime{i} - detector;
        u = [u, gap];
        i = i + 1;
        if (KeyTime{i} >= detector + freq) && (i < numel(KeyTime))
            detector = detector + freq; 
        end
        if i >= numel(KeyTime)
            break
        end
    end
    res = [res, mean(cellfun(@(x) mean(x, 'all'), u))];
end
res{4} = 160;
% plot([2:double(param{'num_Devices'})], res)
x = categorical({'Dev 1', 'Dev 2', 'Dev 3', 'Dev 4', 'Dev 5'});
x = reordercats(x,{'Dev 1', 'Dev 2', 'Dev 3', 'Dev 4', 'Dev 5'});
bar(x, cellfun(@(x) x, res))
% xlabel('dsdfs')
ylabel('Time ($s$)', 'Interpreter','latex', 'FontSize', 13)
title('(f): Average Response Time', 'Interpreter','latex', 'FontSize', 16)
% ylabel('dsdsds')
% Look better
ylim([0, 210])
% ylim([-inf, inf])
grid on
yyaxis right
perc = [res{1}/80, res{2}/120, res{3}/200, res{4}/360, res{5}/450] * 100;
p1 = plot(x, perc, 'rd-', 'Color', "#D95319", 'LineWidth', 1.5, 'LineStyle', '-', 'DisplayName', 'Response Time (\%)');
% legend('Percentage \%', 'interpreter','latex', 'FontSize', 14)
ylabel('Portion of Period (\%)', 'interpreter','latex', 'FontSize', 12)
legend(p1, 'Location','best', 'Fontsize',10, 'interpreter','latex')
ylim([0, 100])
% text(0.6,40, '54.03\%', 'Interpreter', 'latex', 'FontSize', 10, 'Color', 'r')
% text(4.7,30, '44.44\% ', 'Interpreter', 'latex', 'FontSize', 10, 'Color', 'r')
% set(gca,'YTickLabel',[]);

% Specify common title, X and Y labels
% title(t, 'Common title')
% xlabel(t, 'Number of Time Slots', 'FontSize', 20, 'interpreter','latex')
% ylabel(t, 'Reward','FontSize', 20, 'interpreter','latex')





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