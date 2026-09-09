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
xlabel('k', 'FontSize', 15) 
ylabel('$\widehat{L}_p$', 'FontSize', 15, 'Interpreter', 'latex') 
legend([p1, p2], {'Adam', 'SSBFGS'}, 'FontSize', 15)
ax = gca;
ax.FontSize = 15;
hold off
print('loss1-11','-depsc')

figure(2)
p = loglog(iteration2, solution_error);
grid on
p.LineWidth = 1.5;
xlabel('k') 
ylabel('$e_{\sigma}$', 'Interpreter', 'latex') 
ax = gca;
ax.FontSize=15;
print('error-11','-depsc')

figure(3)
p = plot(iteration2, solution_loss2);
grid on
p.LineWidth = 1.5;
xlabel('k') 
ylabel('$\widehat{L}_s$', 'FontSize', 15, 'Interpreter', 'latex') 
ax = gca;
ax.FontSize=15;
print('loss2-11','-depsc')