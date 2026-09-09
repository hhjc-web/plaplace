load('..\solution.mat')

nx=99;
ny=99;

X = -1:2/nx:1;
Y = -1:2/ny:1;

solution = optimal_solution(:,1);
pred = pred_solution(:,1);
solution = reshape(solution, [nx+1, ny+1]);
pred = reshape(pred, [nx+1, ny+1]);

figure(1)
[yy, xx] = meshgrid(X, Y);
R = yy.^2 + xx.^2;
[row,col] = find(R >= 1);
for i=1:length(col)
    solution(row(i),col(i)) = nan;
end
s=surf(xx,yy,solution);
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('true_solution1-12','-depsc')

figure(2)
[yy, xx] = meshgrid(X, Y);
R = yy.^2 + xx.^2;
[row,col] = find(R >= 1);
for i=1:length(col)
    pred(row(i),col(i)) = nan;
end
s=surf(xx,yy,pred);
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('pred_solution1-12','-depsc')

figure(3)
s=surf(xx,yy,abs(pred - solution));
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
caxis([0 5e-4])
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('error_solution1-12','-depsc')

solution = optimal_solution(:,2);
pred = pred_solution(:,2);
solution = reshape(solution, [nx+1, ny+1]);
pred = reshape(pred, [nx+1, ny+1]);

figure(4)
[yy, xx] = meshgrid(X, Y);
R = yy.^2 + xx.^2;
[row,col] = find(R >= 1);
for i=1:length(col)
    solution(row(i),col(i)) = nan;
end
s=surf(xx,yy,solution);
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('true_solution2-12','-depsc')

figure(5)
[yy, xx] = meshgrid(X, Y);
R = yy.^2 + xx.^2;
[row,col] = find(R >= 1);
for i=1:length(col)
    pred(row(i),col(i)) = nan;
end
s=surf(xx,yy,pred);
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('pred_solution2-12','-depsc')

figure(6)
s=surf(xx,yy,abs(pred - solution));
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
caxis([0 5e-4])
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('error_solution2-12','-depsc')

solution = optimal_solution(:,3);
pred = pred_solution(:,3);
solution = reshape(solution, [nx+1, ny+1]);
pred = reshape(pred, [nx+1, ny+1]);

solution(1,1) = solution(1,1) + 0.2;
figure(7)
[yy, xx] = meshgrid(X, Y);
R = yy.^2 + xx.^2;
[row,col] = find(R >= 1);
for i=1:length(col)
    solution(row(i),col(i)) = nan;
end
s=surf(xx,yy,solution);
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
caxis([-2.5 2.5])
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('true_solution3-12','-depsc')

figure(8)
[yy, xx] = meshgrid(X, Y);
R = yy.^2 + xx.^2;
[row,col] = find(R >= 1);
for i=1:length(col)
    pred(row(i),col(i)) = nan;
end
s=surf(xx,yy,pred);
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
caxis([-2.5 2.5])
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('pred_solution3-12','-depsc')

figure(9)
s=surf(xx,yy,abs(pred - solution));
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
caxis([0 5e-4])
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('error_solution3-12','-depsc')

