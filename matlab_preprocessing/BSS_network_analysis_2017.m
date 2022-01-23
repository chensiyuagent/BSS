clear; clc; close all;

%%%%%%%Task 1%%%%%%%
%%%%%%%1)generate original weighted matrix W%%%%%%%
R = readmatrix('July_trip_raw_data_2017.csv');
[b,i,j]=unique(R,'rows');
count = hist(j,unique(j));   
B = [b,count'];
[r,c] = size(B);
temp1 = [];
for k = 1:r
    a = B(k,1);
    b = B(k,2);
    temp1(a,b) = B(k,3);
end
W = temp1; 
[row,col] = size(W);

%%%%%%%2)Threshold matrix decision making%%%%%%%%
% based on paper 'Weight thresholding on complex networks'
T = 0:0.01:0.3; % sample time
output = zeros(length(T),1); % sigma initialization
NoneZ = W(:);
zeroC = find(NoneZ==0);
NoneZ(zeroC) = [];
for j=1:length(T)
	threshold = W; 
	percentile = prctile(abs(NoneZ(:)),T(j)*100); 
	threshold(W < percentile) = 0;
	output(j) = absSpecSim(W, threshold); % sigma
end
figure(1);
plot(T,output(:),'Marker','^');
% generate Laplacian matrix L
d = sum(W,2);
D = [];
sum1 = 0;
avgw1 = 0;
for p = 1:row
    D(p,p) = d(p);
    sum1 = sum1 + d(p);
    avgw1 = sum1/r;
end
L = D - W;
% generate adjacency matrix A
temp2 = [];
%temp3 = [];
for x = 1:row
    for y = 1:col
        if threshold(x,y)~=0
            temp2(x,y) = 1;
        else temp2(x,y) = 0;
        end
    end
end
A = temp2; 
% zeronode = []; withnode = [];
SpaA = sparse(A);
% O = zeros(1,row);
% 
% for o = 1:row
%     if ((A(o,:)==O)||(A(:,o) == O'))
%         zeronode = o;
%     else withnode = o;
%     end
% end

writematrix(A,'2017.csv')
%%%%%3)Visualization and analysis%%%%%%%%
% % G1 is original graph
% G1 = digraph(W);
% [deg1,indeg1,outdeg1] = degrees(W);
% figure(2);
% plot(G1,'Layout','force');
% %  joint probability distribution of indegree and outdegree
% x1 = 0:10:200; 
% y1 = 0:10:200; 
% [X1,Y1] = meshgrid(x1,y1); 
% pdf1 = hist3([indeg1', outdeg1'],{x1 y1}); 
% pdf_normalize1 = (pdf1'./length(indeg1)); 
% figure(3);
% % network density of G1
% surf(X1,Y1,pdf_normalize1);
% netden1 = nnz(adjacency(G1))./numel(adjacency(G1)); 
% fprintf('The original network density is %d\n', netden1);
% [Wavg1, Wavg2, C_w] = clustCoeff(W);
% fprintf("The original network's average cluster coefficient is %d\n", Wavg2);
% 
% % G2 is sparsified graph
% G2 = digraph(A);   
% [deg2,indeg2,outdeg2] = degrees(A);
% figure(4);
% plot(G2,'Layout','force');
% % joint probability distribution of indegree and outdegree
% x2 = 0:10:200; 
% y2 = 0:10:200; 
% [X2,Y2] = meshgrid(x2,y2); 
% pdf2 = hist3([indeg2', outdeg2'],{x2 y2}); 
% pdf_normalize2 = (pdf2'./length(indeg2)); 
% figure(5);
% % network density of G2
% surf(X2,Y2,pdf_normalize2);
% netden2 = nnz(adjacency(G2))./numel(adjacency(G2));
% fprintf('The spasified network density is %d\n', netden2);
% [Aavg1, Aavg2, C_a] = clustCoeff(A);
% fprintf("The spasified network's average cluster coefficient is %d\n", Aavg2);

