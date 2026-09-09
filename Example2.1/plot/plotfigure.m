load('..\solution.mat')

nx=98;
ny=98;

X = -1:2/nx:1;
Y = -1:2/ny:1;

solution = optimal_solution;
pred = pred_solution;
solution = reshape(solution, [nx+1, ny+1]);
pred = reshape(pred, [nx+1, ny+1]);
preddy = pred;

dy = 2/ny;
u0 = zeros(nx+1,1);
pred = zeros(nx+1, ny+1);
pred(:, 1) = u0;
for j = 2:ny+1
    pred(:, j) = pred(:, j-1) + (preddy(:, j-1) + preddy(:, j)) * dy / 2;
end

figure(1)
[yy, xx] = meshgrid(X, Y);
% R = (yy-1).^2 + (xx-1).^2;
% [row,col] = find(R >= 1);
% for i=1:length(col)
%     solution(row(i),col(i)) = nan;
% end
s=surf(xx,yy,solution);
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('true_solution-2','-depsc')

figure(2)
[yy, xx] = meshgrid(X, Y);
% R = (yy-1).^2 + (xx-1).^2;
% [row,col] = find(R >= 1);
% for i=1:length(col)
%     pred(row(i),col(i)) = nan;
% end
s=surf(xx,yy,pred);
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('pred_solution-21','-depsc')

% figure(3)
% s=surf(xx,yy,abs(pred - solution));
% s.EdgeColor = 'none';
% ax = gca;
% ax.YDir = 'normal';
% ax.FontSize=15;
% colorbar();
% set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
% view(2)
% print('error_solution1-21','-depsc')