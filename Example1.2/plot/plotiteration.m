load('..\loss1.mat')
iteration1 = iteration;
solution_loss1 = solution_loss;
load('..\loss2.mat')
iteration2 = iteration;
solution_loss2 = solution_loss;

figure(1)
p1 = loglog(iteration1(1:11), solution_loss1(1:11), 'r', 'LineWidth', 1.5);
hold on
p2 = loglog(iteration1(11:end), solution_loss1(11:end), 'b', 'LineWidth', 1.5);
grid on
xlabel('Iteration', 'FontSize', 12) 
ylabel('Loss', 'FontSize', 12) 
legend([p1, p2], {'Adam', 'SSBFGS'}, 'FontSize', 12)
ax = gca;
ax.FontSize = 12;
hold off
print('loss1-12','-depsc')

figure(2)
p = loglog(iteration2, solution_error);
grid on
p.LineWidth = 1.5;
xlabel('Iteration') 
ylabel('Relative error') 
ax = gca;
ax.FontSize=12;
print('error-12','-depsc')

figure(3)
p = plot(iteration2(1:100), solution_loss2(1:100));
grid on
p.LineWidth = 1.5;
xlabel('Iteration') 
ylabel('Loss') 
ax = gca;
ax.FontSize=12;
print('loss2-12','-depsc')