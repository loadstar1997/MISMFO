 function [M_best_fitness,M_best_pos,Convergence_curve]=MISMFO(pop,Max_iteration,lb,ub,dim,Rd,miu,sigma1,sigma2,fobj)
%% MISMFO algorithm 
%% ======parameter setting ======
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ========= input  =========%%%%%%%%%
% pop : population
% Max_iteration
% lb : lower bound of all optimized parameters
% ub : upper bound of all optimized parameters
% dim : the number of optimized parameters
% Rd: random parameter
% miu: chaos mapping parameter
% sigma1: the number of iterations in the first phase
% sigma2: the number of iterations in the second phase
% fobj:  @YourCostFunction
% ========= output  =========%%%%%%%%
% M_best_fitness :  Optimal function value
% M_best_pos :  Optimal parameter values
% Convergence_curve : The vector corresponding to the value of the convergence curve
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('MISMFO is optimizing your problem');

ub = ub.*ones(1,dim);
lb = lb.*ones(1,dim);  

K1=round(sigma1*Max_iteration);
K2=round(sigma2*Max_iteration);

%% pre-allocation of different variables
M_pos = zeros(pop,dim);  % Moth Location
M_fit=zeros(1,pop); % Moth fitness
F1_pos=zeros(pop,dim); % Mutant flames
Ft_pos=zeros(pop,dim); % Mutant flames in t-th iteration
Ft_fit=zeros(1,pop);   % Fitness of mutant flames
Convergence_curve=zeros(1,Max_iteration); 

%% Logistic Chaos Mapping Initialization
for j=1:dim
    M_pos(1,j)=rand(); 
    for i=2:pop
        M_pos(i,j) =miu*M_pos(i-1,j)*(1-M_pos(i-1,j));
    end
end
M_pos=lb+(ub-lb).*M_pos; 

% Check if moths go out of the search spaceand bring it back
for i=1:pop
        Flag4ub=M_pos(i,:)>ub;
        Flag4lb=M_pos(i,:)<lb;
        M_pos(i,:)=(M_pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        % Calculate the fitness of moths
        M_fit(1,i)=fobj(M_pos(i,:));
end

[sort_M_fit,s]=sort(M_fit);
M_best_pos=M_pos(s(1),:);
M_best_fitness=sort_M_fit(1);

Iteration=1;

%%  Main loop
while Iteration<Max_iteration+1
     % Determine the number of flames for this iteration
     if Iteration<K1
        Flame_no = round(pop-(pop-1)/(Max_iteration*K1^4).*Iteration.^5);
     elseif Iteration>K2
        Flame_no = round(((1-pop)*K2+pop*Max_iteration-Max_iteration)/(K2-Max_iteration)^4/Max_iteration*(Iteration-Max_iteration).^4+1);
     else
        Flame_no=round(pop-Iteration*((pop-1)/Max_iteration));        
     end
     
     % Determine the original flame position in this iteration FO_pos
     if Iteration==1
        % Sort the first population of moths
        [fitness_sorted,I]=sort(M_fit);
        sorted_population=M_pos(I,:);
        
        % Update the flames
        FO_pos=sorted_population;
        FO_fit=fitness_sorted;
     else
         % Sort the moths
        double_population=[M_pos;F_pos];
        double_fitness=[M_fit F_fit];
        
        [double_fitness_sorted, I]=sort(double_fitness);
        double_sorted_population=double_population(I,:);
        
        fitness_sorted=double_fitness_sorted(1:pop);
        sorted_population=double_sorted_population(1:pop,:);
        
        % Update the flames
        FO_pos=sorted_population;
        FO_fit=fitness_sorted;
     end
     %% Generating the current round of mutat flames
     rd=Rd*(1-Iteration/Max_iteration);  % Calculate the perturbation factor for this iteration
     fg=rand*(1-Iteration/Max_iteration)^2;  % Calculate the degree of mutation for this iteration
     n=round(pop*rd);   % Number of flames in this round of mutations
     for i=1:n
         F1_pos(i,:)=FO_pos(pop-n+i,:);   % Flame extraction to be mutated
     end
     % Perform mutation operations
     for i=1:n
         r=rand();
         for j=1:dim
             if r>0.5
                 Ft_pos(i,j)=F1_pos(i,j)+(ub(j)-F1_pos(i,j))*fg;
             else
                 Ft_pos(i,j)=F1_pos(i,j)-(F1_pos(i,j)-lb(j))*fg;
             end
         end
     end
     % Check if mutat moths go out of the search spaceand bring it back
     for i=1:n
        Flag4ub=Ft_pos(i,:)>ub;
        Flag4lb=Ft_pos(i,:)<lb;
        Ft_pos(i,:)=(Ft_pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        % Calculate the fitness of moths
        Ft_fit(1,i)=fobj(Ft_pos(i,:));
     end
     %% Generate the fusion flames F_pos and F_fit for this round
     double_Flame=[FO_pos;Ft_pos(1:n,:)];
     double_Flame_fitness=[FO_fit,Ft_fit(1:n)];
    
    [double_Flame_fitness_sorted,I]=sort(double_Flame_fitness);
    double_sorted_Flame=double_Flame(I,:);
    
    F_fit=double_Flame_fitness_sorted(1:pop);
    F_pos=double_sorted_Flame(1:pop,:);
    
    %% Determine the moths produced in this iteration of the round
    a=-1+Iteration*((-1)/Max_iteration);
    b=1;
    t=(a-1)*rand+1;
    for i=1:pop
        % add adaptative weight 
        M_1=F_fit(1); % Adaptation value of the highest adapted moth, the fitness function is the minimum value function
        M_i=M_fit(i);   % The i-th moth is the adaptation value
        w=0.5*sin(pi.*abs(M_1/M_i)-pi/2)+0.5;
        for j=1:dim
             % first phase
            if Iteration<=K1 %&& Iteration>K2
                if i<=Flame_no
                    distance_to_flame=abs(F_pos(i,j)-M_pos(i,j));
                    M_pos(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+F_pos(i,j);
                else
                    distance_to_flame=abs(F_pos(Flame_no,j)-M_pos(i,j));
                    M_pos(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+F_pos(Flame_no,j);
                end
            else
                 if i<=Flame_no
                     distance_to_flame=abs(F_pos(i,j)-M_pos(i,j));
                     M_pos(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+w*F_pos(i,j)+(1-w)*M_best_pos(j);
                 else
                     distance_to_flame=abs(F_pos(Flame_no,j)-M_pos(i,j));
                     M_pos(i,j)=distance_to_flame*exp(b.*t).*cos(t.*2*pi)+w*F_pos(Flame_no,j)+(1-w)*M_best_pos(j);
                 end
            end
        end
        %  Check if the new moths go out of the search spaceand bring it back
        Flag4ub=M_pos(i,:)>ub;
        Flag4lb=M_pos(i,:)<lb;
        M_pos(i,:)=(M_pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        
        % calculate the fitness of new moths
        M_fit(1,i)=fobj(M_pos(i,:));
    end
    
    [sort_M_fit,s]=sort(M_fit);
    M_best_pos=M_pos(s(1),:);
    M_best_fitness=sort_M_fit(1);   
    
    

    Convergence_curve(Iteration)=F_fit(1);
    Iteration=Iteration+1;
end
% fprintf('The best value of MISMFO algorithm is：%d',F_fit(1))

end
