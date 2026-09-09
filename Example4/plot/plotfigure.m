load('..\solution.mat')

nx=98;
ny=98;

X = -1:2/nx:1;
Y = -1:2/ny:1;

solution = optimal_solution(:,3);
pred = pred_solution(:,3);
solution = reshape(solution, [nx+1, ny+1]);
pred = reshape(pred, [nx+1, ny+1]);

figure(1)
[yy, xx] = meshgrid(X, Y);
% R = yy.^2 + xx.^2;
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
caxis([-1.58 -1.53])
view(2)
print('true_solution-4','-depsc')

figure(2)
[yy, xx] = meshgrid(X, Y);
% R = yy.^2 + xx.^2;
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
caxis([-1.58 -1.53])
print('pred_solution-4','-depsc')

figure(3)
s=surf(xx,yy,abs(pred - solution));
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('error_solution-4','-depsc')
