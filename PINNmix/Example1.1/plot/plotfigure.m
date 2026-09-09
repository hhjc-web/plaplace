load('..\solution1.mat')

nx=99;
ny=99;

X = -1:2/nx:1;
Y = -1:2/ny:1;

%% solution

solution = optimal_solution(:);
pred = pred_solution(:);
solution = reshape(solution, [100, 100]);
% solution = transpose(solution);
pred = reshape(pred, [100, 100]);
% % pred = transpose(pred);

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

figure(2)
[yy, xx] = meshgrid(X, Y);
R = yy.^2 + xx.^2;
[row,col] = find(R > 1);
for i=1:length(col)
    pred(row(i),col(i)) = nan;
end
s=surf(xx,yy,pred);
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
caxis([0 1])
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('pred_solution-11-PINNmix','-depsc')

figure(3)
s=surf(xx,yy,abs(pred - solution));
s.EdgeColor = 'none';
ax = gca;
ax.YDir = 'normal';
ax.FontSize=18;
colorbar();
caxis([0 0.1])
set(gca,'xtick',[],'xticklabel',[])
set(gca,'ytick',[],'yticklabel',[])
view(2)
print('error_solution-11-PINNmix','-depsc')

%% grad_solution_1

% solution = optimal_grad_solution(:,1);
% pred = pred_grad_solution(:,1);
% solution = reshape(solution, [100, 100]);
% solution = transpose(solution);
% pred = reshape(pred, [100, 100]);
% pred = transpose(pred);
% 
% figure(4)
% [yy, xx] = meshgrid(X, Y);
% R = yy.^2 + xx.^2;
% [row,col] = find(R >= 1);
% for i=1:length(col)
%     solution(row(i),col(i)) = nan;
% end
% s=surf(xx,yy,solution);
% s.EdgeColor = 'none';
% ax = gca;
% ax.YDir = 'normal';
% ax.FontSize=15;
% colorbar();
% set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
% view(2)
% 
% figure(5)
% [yy, xx] = meshgrid(X, Y);
% R = yy.^2 + xx.^2;
% [row,col] = find(R > 1);
% for i=1:length(col)
%     pred(row(i),col(i)) = nan;
% end
% s=surf(xx,yy,pred);
% s.EdgeColor = 'none';
% ax = gca;
% ax.YDir = 'normal';
% ax.FontSize=15;
% colorbar();
% set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
% view(2)
% print('pred_solution-11-PINN','-depsc')
% 
% figure(6)
% s=surf(xx,yy,abs(pred - solution));
% s.EdgeColor = 'none';
% ax = gca;
% ax.YDir = 'normal';
% ax.FontSize=15;
% colorbar();
% set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
% view(2)
% print('error_solution-11-PINN','-depsc')

%% grad_solution_2

% solution = optimal_grad_solution(:,2);
% pred = pred_grad_solution(:,2);
% solution = reshape(solution, [101, 101]);
% solution = transpose(solution);
% pred = reshape(pred, [101, 101]);
% pred = transpose(pred);
% 
% figure(7)
% [yy, xx] = meshgrid(X, Y);
% R = yy.^2 + xx.^2;
% [row,col] = find(R >= 1);
% for i=1:length(col)
%     solution(row(i),col(i)) = nan;
% end
% s=surf(xx,yy,solution);
% s.EdgeColor = 'none';
% ax = gca;
% ax.YDir = 'normal';
% ax.FontSize=15;
% colorbar();
% set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
% view(2)
% print('true_gradsolution2-1','-depsc')
% 
% figure(8)
% [yy, xx] = meshgrid(X, Y);
% R = yy.^2 + xx.^2;
% [row,col] = find(R > 1);
% for i=1:length(col)
%     pred(row(i),col(i)) = nan;
% end
% s=surf(xx,yy,pred);
% s.EdgeColor = 'none';
% ax = gca;
% ax.YDir = 'normal';
% ax.FontSize=15;
% colorbar();
% set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
% view(2)
% print('pred_gradsolution2-1','-depsc')
% 
% figure(9)
% s=surf(xx,yy,abs(pred - solution));
% s.EdgeColor = 'none';
% ax = gca;
% ax.YDir = 'normal';
% ax.FontSize=15;
% colorbar();
% set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
% view(2)
% print('error_gradsolution2-1','-depsc')
% 
% %% grad_solution_3
% 
% solution = optimal_grad_solution(:,3);
% pred = pred_grad_solution(:,3);
% solution = reshape(solution, [101, 101]);
% solution = transpose(solution);
% pred = reshape(pred, [101, 101]);
% pred = transpose(pred);
% 
% figure(10)
% [yy, xx] = meshgrid(X, Y);
% R = yy.^2 + xx.^2;
% [row,col] = find(R >= 1);
% for i=1:length(col)
%     solution(row(i),col(i)) = nan;
% end
% s=surf(xx,yy,solution);
% s.EdgeColor = 'none';
% ax = gca;
% ax.YDir = 'normal';
% ax.FontSize=15;
% colorbar();
% set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
% view(2)
% print('true_gradsolution3-1','-depsc')
% 
% figure(11)
% [yy, xx] = meshgrid(X, Y);
% R = yy.^2 + xx.^2;
% [row,col] = find(R > 1);
% for i=1:length(col)
%     pred(row(i),col(i)) = nan;
% end
% s=surf(xx,yy,pred);
% s.EdgeColor = 'none';
% ax = gca;
% ax.YDir = 'normal';
% ax.FontSize=15;
% colorbar();
% set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
% view(2)
% print('pred_gradsolution3-1','-depsc')
% 
% figure(12)
% s=surf(xx,yy,abs(pred - solution));
% s.EdgeColor = 'none';
% ax = gca;
% ax.YDir = 'normal';
% ax.FontSize=15;
% colorbar();
% set(gca,'xtick',[],'xticklabel',[])
% set(gca,'ytick',[],'yticklabel',[])
% view(2)
% print('error_gradsolution3-1','-depsc')