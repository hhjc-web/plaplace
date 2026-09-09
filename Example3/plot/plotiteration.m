load('..\loss.mat')

figure(1)
p = loglog(iteration, solution_error);
grid on
xlim([0,12000])
p.LineWidth = 1.2;
xlabel('Iteration') 
ylabel('Relative error') 
ax = gca;
ax.FontSize=12;
print('error-1','-depsc')

figure(2)
p = plot(iteration, solution_loss);
grid on
p.LineWidth = 1.2;
xlabel('Iteration') 
ylabel('Loss') 
ax = gca;
ax.FontSize=12;
print('loss-1','-depsc')